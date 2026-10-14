from cash_rl import spec
from cash_rl.raycast_env import HallwayRaycastEnv
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest


@pytest.fixture(scope='module')
def env():
    return HallwayRaycastEnv(random_spawn=False, range_noise=False)


def test_gymnasium_api():
    check_env(HallwayRaycastEnv(), skip_render_check=True)


def test_spaces_match_hallway_env_spec(env):
    assert env.observation_space == spec.observation_space()
    assert env.action_space == spec.action_space()


def test_spawn_ranges_match_world_geometry(env):
    # At spawn (0, 0, 0) the corridor walls are 1.2 m to each side in hallway.sdf;
    # SLAM places walls to within about one map cell.
    obs, _ = env.reset(seed=0)
    ranges = obs * spec.MAX_RANGE
    for angle in (-np.pi / 2, np.pi / 2):
        beam = np.argmin(np.abs(spec.OBS_BEAM_ANGLES - angle))
        assert ranges[beam] == pytest.approx(1.2, abs=2 * env.map.resolution)


def test_driving_into_end_wall_ends_episode(env):
    env.reset(seed=0)
    env.pose[:] = (26.5, 0.0, 0.0)
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
        if terminated:
            break
    assert terminated and info['collided']
    assert reward < 0
    # The 28 m end wall should stop the robot's front (plus margin) short of it.
    front = env.pose[0] + spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    assert 27.8 < front < 28.1


def test_random_spawns_are_seeded_and_collision_free():
    env_a, env_b = HallwayRaycastEnv(), HallwayRaycastEnv()
    for seed in range(20):
        _, info_a = env_a.reset(seed=seed)
        _, info_b = env_b.reset(seed=seed)
        np.testing.assert_array_equal(info_a['pose'], info_b['pose'])
        assert not env_a.collides()


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
