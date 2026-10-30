from cash_rl.decision import DecisionLayer
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.scripted import downhill
import numpy as np
import pytest


def encounter(pedestrian_y, spawn_x=19.0, pedestrian_x=27.0, goal_x=27.8):
    """Hallway x 19-28 (no door frames or furniture): one pedestrian walking at the robot."""
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False))
    env.reset(seed=0, options={'spawn': (spawn_x, 0.0, 0.0), 'goal': (goal_x, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[pedestrian_x, pedestrian_y]])
    base._pedestrian_velocities = np.array([[-0.7, 0.0]])
    return env


def drive(env, max_steps=1000):
    """Run downhill (a pure lane follower) to the end; return (info, decided offsets)."""
    offsets = []
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(downhill(env.unwrapped))
        if terminated or truncated:
            return info, offsets
        offsets.append(info['decision_offset'])
    raise AssertionError('episode never ended')


def test_no_pedestrians_keeps_the_center_lane():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, range_noise=False))
    centered = HallwayRaycastEnv(subgoal_lookahead=3.0, range_noise=False)
    options = {'spawn': (8.0, -0.5, 0.0), 'goal': (20.0, 0.0)}
    obs, info = env.reset(seed=0, options=options)
    obs_center, _ = centered.reset(seed=0, options=dict(options, lane_offset=0.0))
    assert info['decision_offset'] == 0.0
    np.testing.assert_allclose(obs, obs_center)


@pytest.mark.parametrize('pedestrian_y, side', [(0.5, -1), (-0.5, 1), (0.1, -1)])
def test_lane_passes_on_the_roomier_side_and_holds_it(pedestrian_y, side):
    # Left of center -> pass right; right of center -> left; near the center (a tie) -> keep
    # right. Dead center in this 2.4 m corridor leaves no lane wide enough on either side.
    env = encounter(pedestrian_y)
    info, offsets = drive(env)
    assert info['is_success'] and not info['collided']
    decided = np.sign([o for o in offsets if o != 0.0])
    assert len(decided) and np.all(decided == side)  # never flips mid-pass


def test_lane_returns_to_center_once_the_pedestrian_is_behind():
    env = encounter(0.5, goal_x=27.8)
    base = env.unwrapped
    seen_offset = False
    for _ in range(1000):
        _, _, terminated, truncated, info = env.step(downhill(base))
        assert not (terminated or truncated)
        seen_offset |= info['decision_offset'] != 0.0
        if base.pedestrians[0, 0] < base.pose[0] - 1.0:
            break
    assert seen_offset and info['decision_offset'] == 0.0


def test_decision_layer_needs_subgoals():
    with pytest.raises(ValueError):
        DecisionLayer(HallwayRaycastEnv())
