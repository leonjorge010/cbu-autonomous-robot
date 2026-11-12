"""Scan-vs-map tracks stand in for the simulator's pedestrian positions."""
from cash_rl.decision import DecisionLayer
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.scripted import downhill
from cash_rl.tracks import ScanDecisionLayer, ScanTracker
import numpy as np


def _place(env, pedestrian):
    env.reset(seed=0, options={'spawn': (8.0, 0.0, 0.0), 'goal': (22.0, 0.0)})
    env.unwrapped.pedestrians = np.array([pedestrian], dtype=np.float64)
    env.unwrapped._pedestrian_velocities = np.array([[-0.7, 0.0]])


def test_a_dense_scan_locates_a_pedestrian_and_its_velocity():
    env = HallwayRaycastEnv(n_pedestrians=1, range_noise=False, subgoal_lookahead=1.5)
    _place(env, [12.0, 0.4])
    tracker = ScanTracker()
    positions, _ = tracker.update(env.unwrapped)
    assert len(positions) == 1
    assert np.hypot(*(positions[0] - env.unwrapped.pedestrians[0])) < 0.45
    env.unwrapped.pedestrians[:] = [[12.0 - 0.07, 0.4]]
    _, velocities = tracker.update(env.unwrapped)
    assert abs(velocities[0, 0] - (-0.7)) < 0.25


def test_map_only_raycast_does_not_stop_on_a_pedestrian():
    env = HallwayRaycastEnv(n_pedestrians=1, range_noise=False)
    _place(env, [10.0, 0.0])
    from cash_rl import spec
    forward = int(np.argmin(np.abs(spec.OBS_BEAM_ANGLES)))
    with_person = env.unwrapped.raycast()
    walls = env.unwrapped.raycast(include_pedestrians=False)
    # The beam aimed at the person stops on them; the map-only cast reaches the wall.
    assert with_person[forward] < 3.0
    assert walls[forward] > with_person[forward] + 0.4


def test_scan_decision_passes_on_the_same_side_as_the_true_positions():
    def make(layer):
        env = layer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1,
                                      range_noise=False))
        env.reset(seed=0, options={'spawn': (19.0, 0.0, 0.0), 'goal': (27.8, 0.0)})
        base = env.unwrapped
        base.pedestrians = np.array([[27.0, 0.5]])
        base._pedestrian_velocities = np.array([[-0.7, 0.0]])
        return env

    privileged, tracked = make(DecisionLayer), make(ScanDecisionLayer)
    agreed = compared = 0
    for _ in range(80):
        for env in (privileged, tracked):
            _, _, terminated, truncated, _ = env.step(downhill(env.unwrapped))
            assert not (terminated or truncated)
        true_offset = privileged.unwrapped.lane_offset
        # lane_offset on the base is what the layer just requested.
        if true_offset == 0.0:
            continue
        compared += 1
        agreed += int(np.sign(tracked.unwrapped.lane_offset) == np.sign(true_offset))
    assert compared >= 10
    assert agreed / compared >= 0.8
