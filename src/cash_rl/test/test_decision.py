from cash_rl.decision import ESCAPE_WAIT, URGENT_LOOKAHEAD, DecisionLayer
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


def test_decision_layer_prob_runs_it_in_that_share_of_episodes():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, subgoal_offset=0.6,
                                          range_noise=False), prob=0.5)
    active = []
    for episode in range(40):
        _, info = env.reset(seed=episode)
        active.append(env.active)
        assert ('decision_offset' in info) == env.active
        _, _, _, _, info = env.step(np.zeros(2))
        assert ('decision_offset' in info) == env.active
    assert 10 < sum(active) < 30
    off = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, subgoal_offset=0.6,
                                          range_noise=False), prob=0.0)
    plain = HallwayRaycastEnv(subgoal_lookahead=3.0, subgoal_offset=0.6, range_noise=False)
    obs, _ = off.reset(seed=3)
    np.testing.assert_allclose(obs, plain.reset(seed=3)[0])  # random lanes untouched


def test_wait_holds_still_when_the_lane_cannot_clear_the_person():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[21.0, 0.0]])
    base._pedestrian_velocities = np.array([[-0.7, 0.0]])
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    assert env._pass_blocked()
    env._waiting = True
    x0 = base.pose[0]
    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert info['waited'] and not (terminated or truncated)
    assert base.pose[0] < x0 + 0.02


def test_a_clear_lane_does_not_wait():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 2.0, base.pose[1]]])
    base._pedestrian_velocities = np.array([[-0.7, 0.0]])
    # The lane point is 0.8 m left of the corridor center, and the person is on
    # that center: more room than the body needs.
    normal = np.array([0.0, 1.0])
    lane_point = base.pose[:2] + 0.8 * normal
    base.near_lane = (lane_point, np.array([1.0, 0.0]), 0.8)
    assert not env._pass_blocked()


def test_a_person_walking_off_the_lane_keeps_the_hold():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 2.0, 0.2]])
    base._pedestrian_velocities = np.array([[0.0, 0.8]])
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    assert env._pass_blocked()
    assert env._escape_action() is None


def test_a_stopped_person_with_room_is_driven_around():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 2.0, 0.0]])
    base._pedestrian_velocities = np.zeros((1, 2))
    env._prev_pedestrians = base.pedestrians.copy()
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    env._waiting = True
    env._wait_time = ESCAPE_WAIT
    # Still straight ahead: turn clear rather than drive into them.
    action = env._escape_action()
    assert action[0] == 0.0 and abs(action[1]) == 1.0
    # Just off the nose but still within 0.8 m to the side: pass, don't floor it.
    base.pose[2] = 0.35
    action = env._escape_action()
    assert abs(float(action[0]) - 0.6) < 1e-5
    # Well clear of them: full speed.
    base.pose[2] = 0.7
    action = env._escape_action()
    assert action[0] == 1.0
    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert info['escaped'] and not info['waited'] and not (terminated or truncated)
    assert base._velocity[0] > 0.0


def test_no_open_side_backs_up():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 2.0, 0.0]])
    base._pedestrian_velocities = np.zeros((1, 2))
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    base._lane_point = lambda point, direction, offset: (base.pedestrians[0].copy(), 0.0, True)
    action = env._escape_action()
    assert action[0] < 0.0
    for _ in range(30):
        assert env._escape_action()[0] < 0.0


def test_a_close_stopped_person_is_backed_away_before_the_turn():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 0.5, 0.0]])
    base._pedestrian_velocities = np.zeros((1, 2))
    env._prev_pedestrians = base.pedestrians.copy()
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    action = env._escape_action()
    assert action[0] < 0.0


def test_a_wall_behind_holds_instead_of_reversing_into_it():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[base.pose[0] + 2.0, 0.0]])
    base._pedestrian_velocities = np.zeros((1, 2))
    env._prev_pedestrians = base.pedestrians.copy()
    base.near_lane = (base.pose[:2].copy(), np.array([1.0, 0.0]), 0.0)
    base._lane_point = lambda point, direction, offset: (base.pedestrians[0].copy(), 0.0, True)
    base._clearance_at = lambda x, y: 0.0
    action = env._escape_action()
    assert action[0] == 0.0


def test_a_crosser_keeps_the_side_chosen_at_the_start():
    env = encounter(-0.5)
    base = env.unwrapped
    base._pedestrian_velocities = np.array([[0.0, 1.2]])
    base.pedestrians = np.array([[base.pose[0] + 1.2, -0.5]])
    env.decider.sides = {}
    env._decide({})
    first = env.decider.sides[0]
    base.pedestrians = np.array([[base.pose[0] + 1.2, 0.6]])
    env._decide({})
    assert env.decider.sides[0] == first


def test_a_close_person_pulls_the_subgoal_in():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=1.5, n_pedestrians=1,
                                          range_noise=False), wait=True)
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[21.0, 0.3]])
    base._pedestrian_velocities = np.array([[-0.7, 0.0]])
    env._decide({})
    assert base.pass_lookahead == URGENT_LOOKAHEAD
    assert np.hypot(*(base.subgoal - base.pose[:2])) < 1.2


def test_a_lane_remains_in_the_last_metres_while_someone_is_ahead():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                          range_noise=False))
    env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (20.2, 0.0)})
    base = env.unwrapped
    base.pedestrians = np.array([[19.8, 0.0]])
    base._pedestrian_velocities = np.array([[-0.6, 0.0]])
    env._decide({})
    assert base.lane_to_goal and base.near_lane is not None


def test_the_lane_still_drops_at_the_goal_with_nobody_ahead():
    env = DecisionLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, range_noise=False))
    _, info = env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (20.2, 0.0)})
    assert env.unwrapped.near_lane is None
    assert info['decision_offset'] == 0.0


def test_a_stale_lane_index_is_dropped_when_a_pedestrian_disappears():
    # The side memory is keyed by list index. A scan track (or a person) that
    # leaves shortens that list; the next step must not read the missing index.
    env = encounter(0.5)
    env.decider.sides[1] = -1.0
    _, _, terminated, truncated, _ = env.step(downhill(env.unwrapped))
    assert not (terminated or truncated)
    assert 1 not in env.decider.sides
