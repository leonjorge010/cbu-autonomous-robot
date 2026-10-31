import json
import os

from cash_rl import spec
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.subgoal_agent import DECISION_STEPS, N_OBS, SubgoalEnv, SubgoalLayer, task_reward
from gymnasium.utils.env_checker import check_env
import numpy as np


def test_gymnasium_api():
    check_env(SubgoalEnv(n_pedestrians=1), skip_render_check=True)


def test_observation_stays_in_bounds_with_pedestrians():
    env = SubgoalEnv(n_pedestrians=2, encounter_prob=1.0)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        assert obs.shape == (N_OBS,) and env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
        if terminated or truncated:
            obs, _ = env.reset()


def test_lane_action_moves_the_lane_and_wait_stops_the_robot():
    env = SubgoalEnv()
    env.reset(seed=0, options={'spawn': (8.0, 0.0, 0.0), 'goal': (20.0, 0.0)})
    for _ in range(3):  # lane 0.5 m left, drive
        env.step(np.array([0.5, -1.0]))
    assert env.base.lane_offset == 0.5 and env.base.near_lane[2] > 0.4
    assert env.base.pose[0] > 8.2
    for _ in range(4):  # wait: brake to a stop and stay
        _, _, _, _, info = env.step(np.array([0.5, 1.0]))
        assert info['waiting']
    stopped = env.base.pose.copy()
    env.step(np.array([0.5, 1.0]))
    np.testing.assert_allclose(env.base.pose, stopped)


def test_one_step_runs_decision_steps_control_steps():
    env = SubgoalEnv()
    env.reset(seed=0, options={'spawn': (8.0, 0.0, 0.0), 'goal': (20.0, 0.0)})
    env.step(np.array([0.0, -1.0]))
    assert env.base._step_count == DECISION_STEPS


def test_task_reward_drops_only_the_lane_penalty():
    assert task_reward(1.0, {'lane_error': 0.6}) == 1.0 + spec.subgoal_lane_penalty(0.6)
    assert task_reward(1.0, {'lane_error': np.nan}) == 1.0
    assert task_reward(1.0, {}) == 1.0


def test_subgoal_layer_runs_a_saved_agent_over_a_motion_policy(tmp_path):
    from stable_baselines3 import PPO
    agent = PPO('MlpPolicy', SubgoalEnv(), n_steps=64, batch_size=64, seed=0, device='cpu')
    agent.save(tmp_path / 'final_model')
    with open(os.path.join(tmp_path, 'config.json'), 'w') as f:
        json.dump({'args': {'algo': 'ppo'}}, f)
    env = SubgoalLayer(HallwayRaycastEnv(subgoal_lookahead=3.0, n_pedestrians=1),
                       str(tmp_path / 'final_model.zip'))
    obs, info = env.reset(seed=0)
    assert obs.shape == (spec.N_OBS,) and 'decision_offset' in info
    offsets = []
    for step in range(1, 3 * DECISION_STEPS + 1):
        obs, _, terminated, truncated, info = env.step(np.array([0.5, 0.0]))
        if terminated or truncated:
            break
        if step % DECISION_STEPS == 0:
            offsets.append(info['decision_offset'])
        if info['waiting']:
            assert abs(env.unwrapped._velocity[0]) <= 0.5
    assert len(offsets) >= 1
