import os

from cash_rl import spec
from cash_rl.occupancy_map import default_map_yaml_in_workspace, keepout_yaml_for
from cash_rl.occupancy_map import load_free_mask
from cash_rl.raycast_env import BRANCH_MIN_Y, ENCOUNTER_DISTANCE_RANGE, ENCOUNTER_LATERAL
from cash_rl.raycast_env import HallwayRaycastEnv, ROUTE_STEP
from cash_rl.raycast_env import PEDESTRIAN_RADIUS, PEDESTRIAN_SLOW_FACTOR
from cash_rl.raycast_env import PEDESTRIAN_SPEED_RANGE, PEDESTRIAN_STOP_DISTANCE
from cash_rl.raycast_env import PEDESTRIAN_WALL_CLEARANCE
from cash_rl.scripted import downhill
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

DOOR_101 = spec.DOOR_GOALS['door_101']
DOOR_202 = spec.DOOR_GOALS['door_202']


@pytest.fixture(scope='module')
def env():
    return HallwayRaycastEnv(range_noise=False)


def drive(env, policy, max_steps=2000):
    """Run one episode with policy(env) -> action; return (total reward, last info)."""
    total = 0.0
    for _ in range(max_steps):
        _, reward, terminated, truncated, info = env.step(policy(env))
        total += reward
        if terminated or truncated:
            return total, info
    raise AssertionError('episode never ended')


def test_gymnasium_api():
    check_env(HallwayRaycastEnv(), skip_render_check=True)


def test_spaces_match_hallway_env_spec(env):
    assert env.observation_space == spec.observation_space()
    assert env.action_space == spec.action_space()


def test_spawn_ranges_match_world_geometry(env):
    # At spawn (0, 0, 0) the corridor walls are 1.2 m to each side in hallway.sdf;
    # SLAM places walls to within about one map cell.
    obs, _ = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    ranges = obs[:spec.N_OBS_BEAMS] * spec.MAX_RANGE  # newest scan
    for angle in (-np.pi / 2, np.pi / 2):
        beam = np.argmin(np.abs(spec.OBS_BEAM_ANGLES - angle))
        assert ranges[beam] == pytest.approx(1.2, abs=2 * env.map.resolution)


def test_goal_observation_is_robot_relative(env):
    obs, _ = env.reset(seed=0, options={'spawn': (0.0, 0.0, np.pi / 2), 'goal': DOOR_101})
    distance, cos_bearing, sin_bearing = obs[-5:-2]
    assert distance == pytest.approx(4.0 / spec.MAX_GOAL_DISTANCE)
    # Facing +y with the goal at +x: the goal is 90 degrees to the right.
    assert cos_bearing == pytest.approx(0.0, abs=1e-6)
    assert sin_bearing == pytest.approx(-1.0)


def test_geodesic_distance_goes_around_the_junction_corner(env):
    _, info = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_202})
    # Straight line is 16.6 m; around the branch's inner corner at (12.8, 1.2) it is
    # ~20.8 m, and the 8-connected grid overestimates by up to ~8%.
    assert 20.3 < info['goal_distance'] < 22.5


@pytest.mark.parametrize('start, goal', [((14.0, 6.0), (20.0, 0.0)), ((4.0, 0.0), DOOR_202)])
def test_cheapest_route_rounds_the_junction_corner_clear_of_collision(env, start, goal):
    # Follow the distance field downhill cell by cell; the wall cost should keep the
    # route well outside the collision boundary (half-width + margin = 0.295 m).
    m = env.map
    field = m.distance_field(*goal).grid
    row, col = m.to_cell(*start)
    clearances = []
    while field[row, col] > 0.1:
        row, col = min(((row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if dr or dc), key=lambda rc: field[rc])
        clearances.append(m.clearance[row, col])
    assert min(clearances) > spec.FOOTPRINT_HALF_WIDTH + spec.COLLISION_MARGIN + 0.1


def test_driving_into_end_wall_ends_episode_with_penalty(env):
    env.reset(seed=0, options={'spawn': (26.5, 0.0, 0.0), 'goal': DOOR_101})
    total, info = drive(env, lambda env: np.array([1.0, 0.0]))
    assert info['collided'] and not info['is_success']
    assert total < -spec.COLLISION_PENALTY + 1.0
    # The 28 m end wall should stop the robot's front (plus margin) short of it.
    front = env.pose[0] + spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    assert 27.8 < front < 28.1


@pytest.mark.parametrize('name', sorted(spec.DOOR_GOALS))
def test_every_door_goal_is_reachable_from_spawn(env, name):
    goal = spec.DOOR_GOALS[name]
    _, start = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': goal})
    total, info = drive(env, downhill)
    assert info['is_success'] and not info['collided']
    # Shaping telescopes: progress totals start distance minus the final distance.
    expected_min = spec.PROGRESS_WEIGHT * (start['goal_distance'] - spec.GOAL_TOLERANCE) \
        + spec.GOAL_REWARD - spec.TIME_PENALTY * spec.episode_step_limit(
            start['goal_distance'])
    assert total > expected_min


def test_reverse_is_slower_than_forward(env):
    assert spec.action_to_cmd([1.0, 0.0])[0] == pytest.approx(spec.MAX_LINEAR_VEL)
    assert spec.action_to_cmd([-1.0, 0.0])[0] == pytest.approx(-spec.MAX_REVERSE_VEL)
    env.reset(seed=0, options={'spawn': (10.0, 0.0, 0.0), 'goal': DOOR_101})
    for _ in range(50):  # 5 s, long past the acceleration ramp
        env.step(np.array([-1.0, 0.0]))
    assert 10.0 - env.pose[0] < 5.0 * spec.MAX_REVERSE_VEL + 0.01


def test_noisy_scans_drop_some_beams():
    env = HallwayRaycastEnv()
    env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    # From spawn every beam hits a wall well inside MAX_RANGE except along the corridor.
    walls = spec.sanitize_ranges(env.raycast()) < spec.MAX_RANGE - 1.0
    dropped = []
    for _ in range(200):
        obs, *_ = env.step(np.zeros(2))
        dropped.append(obs[:spec.N_OBS_BEAMS][walls] == 1.0)
    assert np.mean(dropped) == pytest.approx(spec.BEAM_DROPOUT_PROB, abs=0.01)


def test_footprint_clearance_geometry():
    hl, hw = spec.FOOTPRINT_HALF_LENGTH, spec.FOOTPRINT_HALF_WIDTH
    assert spec.footprint_clearance([[hl + 0.1, 0.0]]) == pytest.approx(0.1)
    assert spec.footprint_clearance([[0.0, -hw - 0.2]]) == pytest.approx(0.2)
    assert spec.footprint_clearance([[hl + 0.3, hw + 0.4]]) == pytest.approx(0.5)
    assert spec.footprint_clearance([[0.1, 0.1], [hl + 1.0, 0.0]]) == 0.0
    assert spec.proximity_penalty(1.0) == 0.0
    assert spec.proximity_penalty(0.0) == pytest.approx(spec.PROXIMITY_WEIGHT)


def test_clearance_matches_corridor_walls(env):
    # Spawn faces +x in the 2.4 m corridor: the footprint's long sides are
    # 1.2 - half-width from the walls (SLAM walls are good to about a cell).
    env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    expected = 1.2 - spec.FOOTPRINT_HALF_WIDTH
    assert env.clearance() == pytest.approx(expected, abs=2 * env.map.resolution)
    # Rotated 90 degrees the half-length faces the walls instead.
    assert env.clearance((0.0, 0.0, np.pi / 2)) == pytest.approx(
        1.2 - spec.FOOTPRINT_HALF_LENGTH, abs=2 * env.map.resolution)


def test_driving_along_a_wall_costs_a_proximity_penalty(env, monkeypatch):
    # Same drive 0.15 m from the north wall, with and without the penalty.
    y = 1.2 - spec.FOOTPRINT_HALF_WIDTH - 0.15

    def drive_along_wall():
        env.reset(seed=0, options={'spawn': (5.0, y, 0.0), 'goal': (20.0, y)})
        steps = [env.step(np.array([1.0, 0.0])) for _ in range(20)]
        assert not any(info['collided'] for *_, info in steps)
        return sum(reward for _, reward, *_ in steps), [info['clearance'] for *_, info in steps]

    with_penalty, clearances = drive_along_wall()
    penalty = sum(spec.proximity_penalty(c) for c in clearances)
    monkeypatch.setattr(spec, 'PROXIMITY_WEIGHT', 0.0)
    without_penalty, _ = drive_along_wall()
    assert max(clearances) < spec.PROXIMITY_DISTANCE
    assert penalty > 1.0
    assert with_penalty == pytest.approx(without_penalty - penalty)


def test_standing_still_times_out(env):
    _, start = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    total, info = drive(env, lambda env: np.zeros(2))
    assert not info['is_success'] and not info['collided']
    steps = spec.episode_step_limit(start['goal_distance'])
    assert total == pytest.approx(-spec.TIME_PENALTY * steps)


def test_random_spawns_and_goals_are_seeded_and_valid():
    env_a, env_b = HallwayRaycastEnv(), HallwayRaycastEnv()
    lo, hi = env_a.goal_distance_range
    for seed in range(10):
        _, info_a = env_a.reset(seed=seed)
        _, info_b = env_b.reset(seed=seed)
        np.testing.assert_array_equal(info_a['pose'], info_b['pose'])
        np.testing.assert_array_equal(info_a['goal'], info_b['goal'])
        assert not env_a.collides()
        # Goals are sampled per cell, so allow a cell's worth of slack.
        assert lo - 0.1 < info_a['goal_distance'] < hi + 0.1


def test_junction_episodes_cross_the_junction():
    env = HallwayRaycastEnv(goal_distance_range=(1.0, 25.0), junction_prob=1.0)
    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert (info['pose'][1] >= BRANCH_MIN_Y) != (info['goal'][1] >= BRANCH_MIN_Y)


def test_junction_prob_zero_keeps_seeded_episodes():
    plain, zero = HallwayRaycastEnv(), HallwayRaycastEnv(junction_prob=0.0)
    for seed in range(5):
        np.testing.assert_array_equal(plain.reset(seed=seed)[1]['goal'],
                                      zero.reset(seed=seed)[1]['goal'])


def test_scan_collision_covers_rear_of_footprint():
    # A return just behind the rear bumper is a collision, even though it is
    # farther from the (forward-mounted) lidar than one just off the side.
    angles = np.array([np.pi])
    behind_rear = spec.FOOTPRINT_HALF_LENGTH + spec.LIDAR_X + 0.02
    assert spec.scan_hits_footprint(np.array([behind_rear]), angles)
    assert not spec.scan_hits_footprint(np.array([behind_rear + spec.COLLISION_MARGIN]), angles)


def test_render_rgb_array():
    env = HallwayRaycastEnv(render_mode='rgb_array')
    env.reset(seed=0)
    image = env.render()
    assert image.shape == (env.map.height, env.map.width, 3)
    assert image.dtype == np.uint8


def scans(obs):
    """Split an observation's scan part into (N_SCAN_STACK, N_OBS_BEAMS), newest first."""
    return obs[:spec.N_SCAN_STACK * spec.N_OBS_BEAMS].reshape(spec.N_SCAN_STACK, -1)


def test_scan_stack_starts_full_and_shifts_newest_first(env):
    obs, _ = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    assert obs.shape == (spec.N_OBS,)
    history = [scans(obs)[0]] * spec.N_SCAN_STACK
    np.testing.assert_array_equal(scans(obs), history)
    for _ in range(4):
        obs, *_ = env.step(np.array([1.0, 0.5]))
        history.insert(0, scans(obs)[0])
        np.testing.assert_array_equal(scans(obs), history[:spec.N_SCAN_STACK])
    np.testing.assert_array_equal(env.observation(), obs)


def test_single_scan_observation_keeps_newest_scan_and_goal(env):
    obs, _ = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    obs, *_ = env.step(np.array([1.0, 0.0]))
    single = spec.single_scan_observation(obs)
    assert single.shape == (spec.SINGLE_SCAN_N_OBS,)
    np.testing.assert_array_equal(single[:spec.N_OBS_BEAMS], scans(obs)[0])
    np.testing.assert_array_equal(single[spec.N_OBS_BEAMS:], obs[-5:])


def test_no_pedestrians_by_default():
    plain, none = HallwayRaycastEnv(), HallwayRaycastEnv(n_pedestrians=0)
    for seed in range(3):
        obs_a, info_a = plain.reset(seed=seed)
        obs_b, info_b = none.reset(seed=seed)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert info_a['pedestrians'].shape == (0, 2)


def pedestrian_env(xy, velocity=(0.0, 0.0), spawn=spec.SPAWN_POSE, pedestrians_yield=False):
    """Env at spawn with its one pedestrian moved to xy; returns (env, obs)."""
    env = HallwayRaycastEnv(range_noise=False, n_pedestrians=1,
                            pedestrians_yield=pedestrians_yield)
    env.reset(seed=0, options={'spawn': spawn, 'goal': DOOR_101})
    env.pedestrians[0] = xy
    env._pedestrian_velocities[0] = velocity
    return env, env._observe()


def test_pedestrian_blocks_beams_at_its_surface():
    env, obs = pedestrian_env((3.0, 0.0))
    ranges = scans(obs)[0] * spec.MAX_RANGE
    forward = np.argmin(np.abs(spec.OBS_BEAM_ANGLES))
    expected = 3.0 - spec.LIDAR_X - PEDESTRIAN_RADIUS
    assert ranges[forward] == pytest.approx(expected, abs=0.01)
    # Beams pointing elsewhere still see the walls.
    side = np.argmin(np.abs(spec.OBS_BEAM_ANGLES - np.pi / 2))
    assert ranges[side] == pytest.approx(1.2, abs=2 * env.map.resolution)


def test_pedestrian_touching_footprint_is_a_collision():
    front = spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    env, _ = pedestrian_env((front + PEDESTRIAN_RADIUS - 0.02, 0.0))
    assert env.collides()
    env.pedestrians[0] = (front + PEDESTRIAN_RADIUS + 0.02, 0.0)
    assert not env.collides()


def test_pedestrian_walks_into_parked_robot_by_default():
    # Robot parked at x=10 on the centerline; pedestrian walking -x straight at it
    # neither slows, sidesteps nor turns back, so the robot has to move.
    env, _ = pedestrian_env((13.0, 0.0), velocity=(-1.0, 0.0), spawn=(10.0, 0.0, 0.0))
    for steps in range(1, 30):
        before = env.pedestrians[0].copy()
        _, reward, terminated, _, info = env.step(np.zeros(2))
        np.testing.assert_allclose(env.pedestrians[0] - before, (-spec.CONTROL_PERIOD, 0.0))
        if terminated:
            break
    assert info['collided'] and info['hit_pedestrian']
    # Waiting is free, but not in its lane, and being walked into costs the collision.
    assert reward == pytest.approx(
        -spec.PEDESTRIAN_COLLISION_PENALTY - spec.PEDESTRIAN_LANE_PENALTY)


def test_pedestrian_steps_around_parked_robot_without_turning_back():
    # Robot parked at x=10 on the centerline; pedestrian walking -x straight at it.
    env, _ = pedestrian_env((13.0, 0.0), velocity=(-1.0, 0.0), spawn=(10.0, 0.0, 0.0),
                            pedestrians_yield=True)
    total, gaps, steps = 0.0, [], 0
    while env.pedestrians[0, 0] > 8.5:
        _, reward, terminated, _, info = env.step(np.zeros(2))
        total += reward
        steps += 1
        gaps.append(info['min_pedestrian_gap'])
        assert not terminated
        assert env._pedestrian_velocities[0, 0] < 0  # never turned back
        assert steps < 100, 'pedestrian never got past the robot'
    assert min(gaps) > 0.0
    assert abs(env.pedestrians[0, 1]) > 0.3  # passed beside the robot, not through it
    # Parked in its lane until it stepped aside: the lane penalty applied.
    assert total < -spec.TIME_PENALTY * steps


def test_pedestrian_waits_when_it_cannot_step_aside():
    # Robot parked across the corridor near the west end, where there is no room
    # to pass: the pedestrian stops short instead of walking into it or turning back.
    env, _ = pedestrian_env((3.0, 0.0), velocity=(-1.0, 0.0), spawn=(0.0, 0.0, np.pi / 2),
                            pedestrians_yield=True)
    env.map.clearance = env.map.clearance.copy()
    row, _ = env.map.to_cell(np.array([0.0]), np.array([0.0]))
    corridor = np.abs(env.map.origin[1] + (np.arange(env.map.height) + 0.5)
                      * env.map.resolution) < 1.5
    env.map.clearance[corridor & (np.arange(env.map.height) != row[0])] = 0.0
    for _ in range(80):
        _, _, terminated, _, info = env.step(np.zeros(2))
        assert not terminated
    assert info['min_pedestrian_gap'] > 0.0
    assert info['min_pedestrian_gap'] < PEDESTRIAN_STOP_DISTANCE + 0.1
    assert env._pedestrian_velocities[0, 0] < 0


def test_pedestrian_slows_only_with_robot_ahead_inside_comfort_zone():
    front = spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN + PEDESTRIAN_RADIUS
    inside = front + (PEDESTRIAN_STOP_DISTANCE + spec.PEDESTRIAN_COMFORT_DISTANCE) / 2
    spawn = (10.0, 0.0, 0.0)
    for velocity, expected_speed in (((-1.0, 0.0), PEDESTRIAN_SLOW_FACTOR),  # toward
                                     ((1.0, 0.0), 1.0)):                     # away
        env, _ = pedestrian_env((10.0 + inside, 0.0), velocity=velocity, spawn=spawn,
                                pedestrians_yield=True)
        before = env.pedestrians[0].copy()
        env.step(np.zeros(2))
        moved = env.pedestrians[0] - before
        assert abs(moved[0]) == pytest.approx(expected_speed * spec.CONTROL_PERIOD)
        # Only the one walking toward the robot (in its path) steps aside.
        assert (moved[1] != 0) == (velocity[0] < 0)
    # Approaching but still outside the comfort zone: full speed, straight on.
    far = 10.0 + front + spec.PEDESTRIAN_COMFORT_DISTANCE + 0.5
    env, _ = pedestrian_env((far, 0.0), velocity=(-1.0, 0.0), spawn=spawn,
                            pedestrians_yield=True)
    env.step(np.zeros(2))
    np.testing.assert_allclose(env.pedestrians[0], (far - spec.CONTROL_PERIOD, 0.0))


def test_pedestrian_beside_robot_path_only_slows():
    # Walking -x in a lane 0.7 m to the side of the robot's centerline: inside the
    # comfort zone but clear of its footprint, so it slows without a sidestep.
    env, _ = pedestrian_env((10.9, 0.7), velocity=(-1.0, 0.0), spawn=(10.0, 0.0, 0.0),
                            pedestrians_yield=True)
    assert env.pedestrian_gaps()[0] < spec.PEDESTRIAN_COMFORT_DISTANCE
    env.step(np.zeros(2))
    assert env.pedestrians[0, 1] == pytest.approx(0.7)
    assert env.pedestrians[0, 0] == pytest.approx(
        10.9 - PEDESTRIAN_SLOW_FACTOR * spec.CONTROL_PERIOD)


def test_driving_into_pedestrian_costs_more_than_a_wall():
    env, _ = pedestrian_env((2.0, 0.0))  # standing still, so it never yields
    total, info = drive(env, lambda env: np.array([1.0, 0.0]))
    assert info['collided'] and info['hit_pedestrian'] and not info['is_success']
    # Progress (< 2 m) can't offset the penalty; the proximity penalty adds to it.
    assert total < -spec.PEDESTRIAN_COLLISION_PENALTY + spec.PROGRESS_WEIGHT * 2.0
    assert spec.PEDESTRIAN_COLLISION_PENALTY > spec.COLLISION_PENALTY


def test_stopping_for_a_pedestrian_is_not_penalized():
    pose, goal = (0.0, 0.0, 0.0), (5.0, 0.0)
    gaps = [0.2]  # inside the comfort zone
    moving_slowly = spec.PEDESTRIAN_WAIT_SPEED * 2
    assert spec.waiting_for_pedestrian(0.0, gaps)
    assert not spec.waiting_for_pedestrian(moving_slowly, gaps)
    assert not spec.waiting_for_pedestrian(0.0, [spec.PEDESTRIAN_COMFORT_DISTANCE + 0.1])
    waiting, _ = spec.step_outcome(4.0, 4.0, pose, goal, False, pedestrian_gaps=gaps,
                                   waiting=True)
    assert waiting == 0.0
    creeping, _ = spec.step_outcome(4.0, 4.0, pose, goal, False, pedestrian_gaps=gaps)
    assert creeping == pytest.approx(
        -spec.TIME_PENALTY - spec.pedestrian_proximity_penalty(gaps))
    # Still pays the wall proximity penalty, and a collision is never excused.
    assert spec.step_outcome(4.0, 4.0, pose, goal, False, clearance=0.0, waiting=True)[0] < 0
    assert spec.step_outcome(4.0, 4.0, pose, goal, True, pedestrian_gaps=[-0.1],
                             hit_pedestrian=True, waiting=True)[0] \
        == pytest.approx(-spec.PEDESTRIAN_COLLISION_PENALTY)


def test_lane_time_counts_only_pedestrians_closing_on_the_robot_in_their_lane():
    spawn = (10.0, 0.0, 0.0)  # parked, facing +x
    front = spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    # Walking -x at 1 m/s straight at the robot from 3 m: reaches its front edge
    # after (3 - front - radius) s.
    env, _ = pedestrian_env((13.0, 0.0), velocity=(-1.0, 0.0), spawn=spawn)
    assert env.pedestrian_lane_times()[0] == pytest.approx(3.0 - front - PEDESTRIAN_RADIUS)
    # Same, but in a lane beside the robot, or walking away from it: no lane time.
    env, _ = pedestrian_env((13.0, 0.9), velocity=(-1.0, 0.0), spawn=spawn)
    assert env.pedestrian_lane_times()[0] == np.inf
    env, _ = pedestrian_env((13.0, 0.0), velocity=(1.0, 0.0), spawn=spawn)
    assert env.pedestrian_lane_times()[0] == np.inf
    # Coming up from behind: timed at the closing speed, and never if the robot
    # drives away faster than the pedestrian walks.
    env, _ = pedestrian_env((8.0, 0.0), velocity=(0.9, 0.0), spawn=spawn)
    gap = 2.0 - front - PEDESTRIAN_RADIUS
    env._velocity[0] = 0.5
    assert env.pedestrian_lane_times()[0] == pytest.approx(gap / 0.4)
    env._pedestrian_velocities[0] = (0.4, 0.0)
    assert env.pedestrian_lane_times()[0] == np.inf


def test_lane_penalty_ramps_up_to_contact_and_rewards_stepping_aside():
    horizon, full = spec.PEDESTRIAN_LANE_HORIZON, spec.PEDESTRIAN_LANE_PENALTY
    assert spec.pedestrian_lane_penalty([]) == 0.0
    assert spec.pedestrian_lane_penalty([np.inf, horizon, horizon + 1]) == 0.0
    assert spec.pedestrian_lane_penalty([horizon / 2]) == pytest.approx(full / 2)
    assert spec.pedestrian_lane_penalty([0.0, 0.0]) == pytest.approx(2 * full)
    # Waiting doesn't excuse it.
    pose, goal = (0.0, 0.0, 0.0), (5.0, 0.0)
    reward, _ = spec.step_outcome(4.0, 4.0, pose, goal, False, waiting=True,
                                  pedestrian_lane_times=[0.0])
    assert reward == pytest.approx(-full)
    # In the env: in the lane of an oncoming pedestrian costs, beside it doesn't.
    env, _ = pedestrian_env((13.0, 0.0), velocity=(-0.5, 0.0), spawn=(11.0, 0.0, 0.0))
    _, in_lane, _, _, info = env.step(np.zeros(2))
    assert info['min_pedestrian_lane_time'] < horizon
    env, _ = pedestrian_env((13.0, 0.9), velocity=(-0.5, 0.0), spawn=(11.0, 0.0, 0.0))
    _, beside, _, _, info = env.step(np.zeros(2))
    assert info['min_pedestrian_lane_time'] == np.inf
    assert in_lane < beside


def test_wall_collision_is_not_a_pedestrian_hit():
    env = HallwayRaycastEnv(range_noise=False, n_pedestrians=2)
    env.reset(seed=0, options={'spawn': (26.5, 0.0, 0.0), 'goal': DOOR_101})
    env.pedestrians[:] = (5.0, 0.0)
    env._pedestrian_velocities[:] = 0.0
    _, info = drive(env, lambda env: np.array([1.0, 0.0]))
    assert info['collided'] and not info['hit_pedestrian']


def test_pedestrian_proximity_penalty_ramps_to_contact():
    comfort = spec.PEDESTRIAN_COMFORT_DISTANCE
    full = spec.PEDESTRIAN_PROXIMITY_PENALTY
    assert spec.pedestrian_proximity_penalty([]) == 0.0
    assert spec.pedestrian_proximity_penalty([comfort, comfort + 1]) == 0.0
    assert spec.pedestrian_proximity_penalty([comfort / 2]) == pytest.approx(full / 2)
    assert spec.pedestrian_proximity_penalty([0.0, -0.1]) == pytest.approx(2 * full)
    # Without pedestrian arguments step_outcome is unchanged (HallwayEnv's case).
    pose, goal = (0.0, 0.0, 0.0), (5.0, 0.0)
    assert spec.step_outcome(5.0, 4.0, pose, goal, True)[0] == pytest.approx(
        spec.PROGRESS_WEIGHT - spec.TIME_PENALTY - spec.COLLISION_PENALTY)
    reward, _ = spec.step_outcome(5.0, 4.0, pose, goal, True,
                                  pedestrian_gaps=[-0.1], hit_pedestrian=True)
    assert reward == pytest.approx(
        spec.PROGRESS_WEIGHT - spec.TIME_PENALTY - full - spec.PEDESTRIAN_COLLISION_PENALTY)


def test_pedestrians_walk_the_corridors_clear_of_walls():
    env = HallwayRaycastEnv(n_pedestrians=6)
    for seed in range(3):
        _, info = env.reset(seed=seed)
        speeds = np.hypot(*env._pedestrian_velocities.T)
        assert np.all((speeds >= PEDESTRIAN_SPEED_RANGE[0])
                      & (speeds <= PEDESTRIAN_SPEED_RANGE[1]))
        # Corridor-aligned: x motion in the main corridor, y motion in the branch.
        in_branch = env.pedestrians[:, 1] >= BRANCH_MIN_Y
        moving = np.abs(env._pedestrian_velocities)
        assert np.all(moving[in_branch, 0] == 0) and np.all(moving[~in_branch, 1] == 0)
        assert np.all(np.hypot(*(env.pedestrians - info['pose'][:2]).T) >= 2.0)
        moves = np.zeros(len(env.pedestrians))
        for _ in range(300):
            before = env.pedestrians.copy()
            env.step(np.zeros(2))  # robot may be hit; pedestrians keep walking regardless
            moves += np.any(env.pedestrians != before, axis=1)
            clearance = env._clearance_at(env.pedestrians[:, 0], env.pedestrians[:, 1])
            assert np.all(clearance >= PEDESTRIAN_WALL_CLEARANCE)
        # Only turnaround steps stand still, so nobody gets stuck against a wall.
        assert np.all(moves > 250)


def test_pedestrian_episodes_are_seeded():
    env_a, env_b = HallwayRaycastEnv(n_pedestrians=3), HallwayRaycastEnv(n_pedestrians=3)
    for seed in range(3):
        _, info_a = env_a.reset(seed=seed)
        _, info_b = env_b.reset(seed=seed)
        np.testing.assert_array_equal(info_a['pedestrians'], info_b['pedestrians'])
        assert len(info_a['pedestrians']) == 3


def test_route_follows_the_field_to_the_goal_clear_of_walls(env):
    env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_202})
    route = env.route()
    assert np.hypot(*(route[-1] - DOOR_202)) <= ROUTE_STEP
    np.testing.assert_allclose(np.hypot(*np.diff(route, axis=0).T), ROUTE_STEP)
    # Same wall-shy route as the progress reward: around the junction corner.
    clearance = env._clearance_at(route[:, 0], route[:, 1])
    assert clearance.min() > spec.FOOTPRINT_HALF_WIDTH + spec.COLLISION_MARGIN + 0.1
    length = ROUTE_STEP * (len(route) - 1)
    assert length == pytest.approx(env._goal_distance, rel=0.15)


def route_in_own_corridor(env):
    """(route, travelled) up to where the robot's route first leaves its corridor."""
    route = env.route()
    in_branch = route[:, 1] >= BRANCH_MIN_Y
    leaves = np.nonzero(in_branch != in_branch[0])[0]
    route = route[:leaves[0]] if len(leaves) else route
    return route, np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])


def test_encounter_pedestrian_starts_on_the_route_walking_at_the_robot():
    env = HallwayRaycastEnv(n_pedestrians=2, encounter_prob=1.0, goal_distance_range=(8, 25))
    lo, hi = ENCOUNTER_DISTANCE_RANGE
    checked = 0
    for seed in range(20):
        _, info = env.reset(seed=seed)
        route, travelled = route_in_own_corridor(env)
        if travelled[-1] < lo:
            continue  # no room: a random start (test_short_routes_fall_back...)
        checked += 1
        ped, velocity = env.pedestrians[0], env._pedestrian_velocities[0]
        offsets = np.hypot(*(route - ped).T)
        i = int(np.argmin(offsets))
        assert offsets[i] <= ENCOUNTER_LATERAL + ROUTE_STEP
        assert lo - ROUTE_STEP <= travelled[i] <= hi + ROUTE_STEP
        direction = route[min(i + 1, len(route) - 1)] - route[max(i - 1, 0)]
        assert velocity @ direction < 0  # walking back along the route
        assert np.hypot(*(ped - info['pose'][:2])) >= 2.0
        assert env._clearance_at(*ped) >= PEDESTRIAN_WALL_CLEARANCE
        assert len(env.pedestrians) == 2  # the rest start at random as usual
    assert checked >= 14


def test_robot_ignoring_an_encounter_pedestrian_meets_it():
    # Over half of random pedestrian starts never come within 2 m of the robot;
    # encounter starts should nearly always reach the comfort zone.
    env = HallwayRaycastEnv(n_pedestrians=1, encounter_prob=1.0, goal_distance_range=(8, 25))
    met = checked = 0
    for seed in range(20):
        env.reset(seed=seed)
        if route_in_own_corridor(env)[1][-1] < ENCOUNTER_DISTANCE_RANGE[0]:
            continue
        checked += 1
        closest = np.inf
        while True:
            _, _, terminated, truncated, info = env.step(downhill(env))
            closest = min(closest, info['min_pedestrian_gap'])
            if terminated or truncated:
                break
        met += closest < spec.PEDESTRIAN_COMFORT_DISTANCE
    assert checked >= 14 and met == checked


@pytest.mark.parametrize('spawn, goal', [
    (spec.SPAWN_POSE, (2.0, 0.0)),  # route too short
    ((14.0, 4.0, -np.pi / 2), (24.0, 0.0)),  # only 2.8 m of branch before the corner
])
def test_short_routes_fall_back_to_random_pedestrian_starts(spawn, goal):
    env = HallwayRaycastEnv(n_pedestrians=1, encounter_prob=1.0)
    _, info = env.reset(seed=0, options={'spawn': spawn, 'goal': goal})
    assert env._sample_encounter() is None
    assert len(info['pedestrians']) == 1
    assert np.hypot(*(info['pedestrians'][0] - info['pose'][:2])) >= 2.0


def test_encounter_prob_zero_keeps_seeded_episodes():
    plain = HallwayRaycastEnv(n_pedestrians=2)
    zero = HallwayRaycastEnv(n_pedestrians=2, encounter_prob=0.0)
    for seed in range(5):
        np.testing.assert_array_equal(plain.reset(seed=seed)[1]['pedestrians'],
                                      zero.reset(seed=seed)[1]['pedestrians'])


def test_gymnasium_api_with_pedestrians():
    check_env(HallwayRaycastEnv(n_pedestrians=3), skip_render_check=True)


def test_avoider_starts_from_rest_facing_the_goal():
    # Regression: its arrival estimate ignored acceleration, so stopping tied with
    # driving on and a robot at rest facing its goal never left the spawn.
    from cash_rl.scripted import avoider
    env = HallwayRaycastEnv(n_pedestrians=1)
    env.reset(seed=0, options={'spawn': spec.SPAWN_POSE, 'goal': DOOR_101})
    for _ in range(20):
        env.step(avoider(env))
    assert env.pose[0] > 0.5


def test_dagger_numpy_actor_matches_sb3_predict():
    from cash_rl.dagger import actor_action, actor_weights
    from stable_baselines3 import PPO
    env = HallwayRaycastEnv()
    model = PPO('MlpPolicy', env, policy_kwargs={'net_arch': [256, 256]}, seed=0,
                device='cpu')
    weights = actor_weights(model.policy)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        expected, _ = model.predict(obs, deterministic=True)
        assert np.allclose(actor_action(weights, obs), expected, atol=1e-5)
        obs, *_ = env.step(expected)


def test_dagger_collect_labels_every_step_with_the_teacher():
    from cash_rl import dagger
    from cash_rl.scripted import avoider
    dagger._init_worker({'n_pedestrians': 1, 'encounter_prob': 1.0})
    observations, labels, _ = dagger.collect((7, 30, 1.0, None))
    assert observations.shape == (30, spec.N_OBS) and labels.shape == (30, 2)
    # beta 1 / no student: the teacher drives, so replaying its labels retraces the episode.
    env = HallwayRaycastEnv(n_pedestrians=1, encounter_prob=1.0)
    obs, _ = env.reset(seed=7)
    for i in range(30):
        assert np.allclose(obs, observations[i]) and np.allclose(avoider(env), labels[i])
        obs, _, terminated, truncated, _ = env.step(labels[i])
        if terminated or truncated:
            break


@pytest.mark.parametrize('offset', [-0.15, 0.15])
def test_avoider_keeps_right_of_an_oncoming_pedestrian(offset):
    # Whichever side of the robot's line the pedestrian walks, it passes on the
    # robot's left, so a student policy sees one consistent answer.
    from cash_rl.scripted import avoider
    env = HallwayRaycastEnv(n_pedestrians=1, range_noise=False)
    env.reset(seed=0, options={'spawn': (3.0, 0.0, 0.0), 'goal': (12.0, 0.0)})
    env.pedestrians[:] = [[8.0, offset]]
    env._pedestrian_velocities[:] = [[-0.7, 0.0]]
    closest = (np.inf, 0.0)
    for _ in range(150):
        dx, dy = env.pedestrians[0] - env.pose[:2]
        if np.hypot(dx, dy) < closest[0]:
            closest = (np.hypot(dx, dy), dy)
        _, _, terminated, truncated, info = env.step(avoider(env))
        assert not info['collided']
        if terminated or truncated or env.pedestrians[0, 0] < env.pose[0] - 1.0:
            break
    assert closest[1] > 0.0  # pedestrian on the robot's left (+y while heading +x)


TEGR_MAP = os.path.join(os.path.dirname(default_map_yaml_in_workspace()), 'cbu_floor2.yaml')


@pytest.fixture(scope='module')
def tegr_env():
    return HallwayRaycastEnv(map_yaml=TEGR_MAP, n_pedestrians=4, encounter_prob=0.5)


def test_tegr_spawns_goals_and_pedestrians_stay_out_of_keepout_zones(tegr_env):
    allowed = load_free_mask(keepout_yaml_for(TEGR_MAP))[1]
    for seed in range(30):
        tegr_env.reset(seed=seed)
        for x, y in [tegr_env.pose[:2], tegr_env.goal, *tegr_env.pedestrians]:
            row, col = tegr_env.map.to_cell(x, y)
            assert allowed[row, col]


@pytest.mark.parametrize('xy, along_y', [
    ((30.0, 12.0), False),  # south corridor
    ((30.0, 60.0), False),  # north corridor
    ((13.0, 30.0), True),   # west corridor
    ((14.0, 22.0), True),   # west corridor, at TEGR 230's door recess
])
def test_tegr_pedestrians_walk_along_their_corridor(tegr_env, xy, along_y):
    assert (tegr_env._walking_axis(np.array(xy))[1] > 0) == along_y


def test_tegr_has_no_hallway_doors_or_junction():
    env = HallwayRaycastEnv(map_yaml=TEGR_MAP)
    assert env.door_goals == {} and env.spawn_pose is None
    with pytest.raises(ValueError):
        HallwayRaycastEnv(map_yaml=TEGR_MAP, junction_prob=0.3)


def test_limited_distance_field_matches_the_full_one_within_the_limit(env):
    full = env.map.distance_field(5.0, 0.0).grid
    limited = env.map.distance_field(5.0, 0.0, limit=6.0).grid
    near = full <= 6.0
    assert np.allclose(limited[near], full[near]) and np.all(np.isinf(limited[full > 6.1]))


def test_subgoal_sits_lookahead_metres_along_the_route_and_only_changes_the_obs():
    plain, sub = HallwayRaycastEnv(), HallwayRaycastEnv(subgoal_lookahead=3.0)
    options = {'spawn': spec.SPAWN_POSE, 'goal': DOOR_202}
    obs_plain, _ = plain.reset(seed=0, options=options)
    obs_sub, info = sub.reset(seed=0, options=options)
    route = sub.route()
    np.testing.assert_allclose(info['subgoal'], route[int(round(3.0 / ROUTE_STEP))])
    distance, bearing = spec.goal_in_robot_frame(sub.pose, sub.subgoal)
    start = spec.N_SCAN_STACK * spec.N_OBS_BEAMS
    goal_slice = slice(start, start + 3)
    np.testing.assert_allclose(obs_sub[goal_slice], [distance / spec.MAX_GOAL_DISTANCE,
                                                     np.cos(bearing), np.sin(bearing)], atol=1e-6)
    np.testing.assert_array_equal(obs_sub[:goal_slice.start], obs_plain[:goal_slice.start])
    # Same actions, same reward and outcome: only what the policy sees changes.
    while True:
        action = downhill(plain)
        _, r_plain, term_plain, trunc_plain, _ = plain.step(action)
        _, r_sub, term_sub, trunc_sub, info = sub.step(action)
        assert r_sub == r_plain and (term_sub, trunc_sub) == (term_plain, trunc_plain)
        if term_plain or trunc_plain:
            break
    assert info['is_success']


def test_subgoal_is_the_goal_once_it_is_closer_than_the_lookahead():
    env = HallwayRaycastEnv(subgoal_lookahead=3.0)
    _, info = env.reset(seed=0, options={'spawn': spec.SPAWN_POSE,
                                         'goal': np.array(spec.SPAWN_POSE[:2]) + [2.0, 0.0]})
    np.testing.assert_array_equal(info['subgoal'], info['goal'])
    _, info = HallwayRaycastEnv().reset(seed=0, options={'spawn': spec.SPAWN_POSE,
                                                         'goal': DOOR_202})
    np.testing.assert_array_equal(info['subgoal'], info['goal'])
