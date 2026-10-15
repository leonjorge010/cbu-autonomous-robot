from cash_rl import spec
from cash_rl.raycast_env import HallwayRaycastEnv
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
    ranges = obs[:spec.N_OBS_BEAMS] * spec.MAX_RANGE
    for angle in (-np.pi / 2, np.pi / 2):
        beam = np.argmin(np.abs(spec.OBS_BEAM_ANGLES - angle))
        assert ranges[beam] == pytest.approx(1.2, abs=2 * env.map.resolution)


def test_goal_observation_is_robot_relative(env):
    obs, _ = env.reset(seed=0, options={'spawn': (0.0, 0.0, np.pi / 2), 'goal': DOOR_101})
    distance, cos_bearing, sin_bearing = obs[spec.N_OBS_BEAMS:spec.N_OBS_BEAMS + 3]
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


def test_reverse_is_slower_than_forward(env):
    assert spec.action_to_cmd([1.0, 0.0])[0] == pytest.approx(spec.MAX_LINEAR_VEL)
    assert spec.action_to_cmd([-1.0, 0.0])[0] == pytest.approx(-spec.MAX_REVERSE_VEL)
    env.reset(seed=0, options={'spawn': (10.0, 0.0, 0.0), 'goal': DOOR_101})
    for _ in range(50):  # 5 s, long past the acceleration ramp
        env.step(np.array([-1.0, 0.0]))
    assert 10.0 - env.pose[0] < 5.0 * spec.MAX_REVERSE_VEL + 0.01


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
    from cash_rl.raycast_env import BRANCH_MIN_Y
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
