from cash_rl import spec
from cash_rl.raycast_env import BRANCH_MIN_Y, HallwayRaycastEnv
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


def pedestrian_env(xy, velocity=(0.0, 0.0), spawn=spec.SPAWN_POSE):
    """Env at spawn with its one pedestrian moved to xy; returns (env, obs)."""
    env = HallwayRaycastEnv(range_noise=False, n_pedestrians=1)
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


def test_pedestrian_steps_around_parked_robot_without_turning_back():
    # Robot parked at x=10 on the centerline; pedestrian walking -x straight at it.
    env, _ = pedestrian_env((13.0, 0.0), velocity=(-1.0, 0.0), spawn=(10.0, 0.0, 0.0))
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
    # Passing close cost the robot more than the time penalty alone.
    assert total < -spec.TIME_PENALTY * steps - 0.5


def test_pedestrian_waits_when_it_cannot_step_aside():
    # Robot parked across the corridor near the west end, where there is no room
    # to pass: the pedestrian stops short instead of walking into it or turning back.
    env, _ = pedestrian_env((3.0, 0.0), velocity=(-1.0, 0.0), spawn=(0.0, 0.0, np.pi / 2))
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
        env, _ = pedestrian_env((10.0 + inside, 0.0), velocity=velocity, spawn=spawn)
        before = env.pedestrians[0].copy()
        env.step(np.zeros(2))
        moved = env.pedestrians[0] - before
        assert abs(moved[0]) == pytest.approx(expected_speed * spec.CONTROL_PERIOD)
        # Only the one walking toward the robot (in its path) steps aside.
        assert (moved[1] != 0) == (velocity[0] < 0)
    # Approaching but still outside the comfort zone: full speed, straight on.
    far = 10.0 + front + spec.PEDESTRIAN_COMFORT_DISTANCE + 0.5
    env, _ = pedestrian_env((far, 0.0), velocity=(-1.0, 0.0), spawn=spawn)
    env.step(np.zeros(2))
    np.testing.assert_allclose(env.pedestrians[0], (far - spec.CONTROL_PERIOD, 0.0))


def test_pedestrian_beside_robot_path_only_slows():
    # Walking -x in a lane 0.7 m to the side of the robot's centerline: inside the
    # comfort zone but clear of its footprint, so it slows without a sidestep.
    env, _ = pedestrian_env((10.9, 0.7), velocity=(-1.0, 0.0), spawn=(10.0, 0.0, 0.0))
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
    assert spec.step_outcome(5.0, 4.0, pose, goal, True, [-0.1], True)[0] == pytest.approx(
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


def test_gymnasium_api_with_pedestrians():
    check_env(HallwayRaycastEnv(n_pedestrians=3), skip_render_check=True)
