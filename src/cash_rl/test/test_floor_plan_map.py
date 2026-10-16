"""HallwayRaycastEnv on the TEGR 2nd floor map: sidecars, keep-out zones, lookahead."""
import os

from cash_rl import spec
from cash_rl.occupancy_map import default_map_yaml, OccupancyMap
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.scripted import downhill
import numpy as np
import pytest

MAP_YAML = os.path.join(os.path.dirname(default_map_yaml()), 'cbu_floor2.yaml')
pytestmark = pytest.mark.skipif(not os.path.isfile(MAP_YAML), reason=f'no {MAP_YAML}')


@pytest.fixture(scope='module')
def env():
    return HallwayRaycastEnv(map_yaml=MAP_YAML, range_noise=False)


def drive(env, policy, max_steps=3000):
    for _ in range(max_steps):
        _, reward, terminated, truncated, info = env.step(policy(env))
        if terminated or truncated:
            return reward, info
    raise AssertionError('episode never ended')


def test_sidecars_load():
    m = OccupancyMap(MAP_YAML)
    assert m.spawn is not None and len(m.goals) == 26
    assert m.keepout.sum() * m.resolution ** 2 > 150  # 3 learning zones + Stair 1
    assert not (m.traversable & m.keepout).any()
    assert not m.in_keepout(*m.spawn[:2])
    for name, (x, y) in m.goals.items():
        assert m.allowed[m.to_cell(x, y)], name


def test_hallway_map_has_no_keepout():
    assert not OccupancyMap().keepout.any()


def test_random_spawns_and_goals_avoid_keepout(env):
    for seed in range(100):
        _, info = env.reset(seed=seed)
        assert not env.map.in_keepout(*info['pose'][:2])
        assert not env.map.in_keepout(*info['goal'])


def test_entering_a_keepout_zone_ends_the_episode_as_a_collision(env):
    # West corridor, beside Learning Zone 3's opening (south of its counter), facing in.
    env.reset(seed=0, options={'spawn': (14.1, 35.0, 0.0), 'goal': env.map.goals['tegr_201']})
    reward, info = drive(env, lambda _: np.array([1.0, 0.0]), max_steps=100)
    assert info['keepout'] and info['collided'] and not info['is_success']
    assert reward < -spec.COLLISION_PENALTY + 1
    assert env.map.in_keepout(*info['pose'][:2])


def test_waypoint_is_lookahead_along_the_route(env):
    field = env.map.distance_field(*env.map.goals['tegr_221'])
    start = env.map.spawn[:2]
    waypoint = field.waypoint(*start, lookahead=10.0)
    # Route distance drops by about the lookahead (the wall cost adds a little).
    assert 9.0 < field(*start) - field(*waypoint) < 12.0
    assert env.map.traversable[env.map.to_cell(*waypoint)]
    # Within the lookahead of the goal, the waypoint is the goal.
    near = field.waypoint(*env.map.goals['tegr_221'], lookahead=10.0)
    assert np.allclose(near, env.map.goals['tegr_221'])


def test_lookahead_keeps_the_observed_goal_in_training_range():
    env = HallwayRaycastEnv(map_yaml=MAP_YAML, range_noise=False, lookahead=10.0)
    obs, info = env.reset(seed=0, options={'spawn': env.map.spawn,
                                           'goal': env.map.goals['tegr_221']})
    assert info['goal_distance'] > 40
    assert obs[spec.N_OBS_BEAMS] * spec.MAX_GOAL_DISTANCE < 10.5
    assert not np.allclose(env.observed_goal, env.goal)


@pytest.mark.parametrize('name', ['tegr_221', 'tegr_204', 'elevator_2', 'tegr_229'])
def test_far_goals_are_reachable_around_the_keepout_zones(env, name):
    env.reset(seed=0, options={'spawn': env.map.spawn, 'goal': env.map.goals[name]})
    _, info = drive(env, downhill)
    assert info['is_success'] and not info['collided']
