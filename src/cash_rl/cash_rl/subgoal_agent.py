r"""Learned subgoal agent: the upper half of the hierarchy, trained with RL.

    ros2 run cash_rl train_subgoal --motion runs/<motion run>/final_model.zip --map cbu_floor2 \
        --pedestrians 1 --encounter-prob 0.7 --name tegr_subgoal_s0
    ros2 run cash_rl evaluate runs/<motion run>/final_model.zip --map cbu_floor2 --random 200 \
        --pedestrians 1 --encounter-prob 1 --subgoal-agent runs/tegr_subgoal_s0/final_model.zip

Every DECISION_STEPS control steps (0.5 s) it picks, for the motion agent below
it (a frozen lane-following policy, or scripted.downhill):
  - a lane: action[0] * MAX_LANE_OFFSET m left (+) / right (-) of the corridor
    center, which HallwayRaycastEnv clips to the room there and moves the
    motion agent's subgoal into (the planner still picks how far along the route it lies);
  - whether to wait: action[1] > 0 stops the robot instead of running the
    motion agent, so it can let a pedestrian pass where no lane is wide enough.

It sees only what the robot will have: LiDAR (the observed beams at now, 0.5 s
and 1 s ago, so pedestrians show as moving returns), the planner's next
subgoal on the corridor center and the end goal in the robot frame, the room
for a lane on each side (from the map), the lane in use, the robot's velocity
and its previous action. No pedestrian positions.

Its reward is the task reward summed over its control steps, without the
motion agent's lane penalty (which only grades how well the motion agent
follows the lane this agent chose): progress, goal, collisions (pedestrian
hits cost double), pedestrian proximity and the pedestrian lane penalty.

SubgoalEnv is the training env (one step = one decision). SubgoalLayer runs a
trained agent on top of any motion policy for evaluate/view_env, like
decision.DecisionLayer does for the rule-based layer.
"""
import argparse
import datetime
import json
import os

from cash_rl import spec
from cash_rl.occupancy_map import default_map_yaml, use_map
from cash_rl.raycast_env import HallwayRaycastEnv
import gymnasium as gym
import numpy as np

DECISION_STEPS = 5  # control steps (0.1 s) per decision
# Scans kept for the observation: now, and this many decisions back.
SCAN_HISTORY = (0, 1, 2)
MAX_LANE_OFFSET = 1.0  # m; the env clips it to the room the corridor has
# m: how far ahead on the route the planner's next subgoal is observed (this
# agent's view), and the motion subgoal distance for downhill. A model motion
# agent keeps the lookahead it was trained with.
LOOKAHEAD = 3.0
ROOM_SCALE = 2.0  # m, lane room normalization
N_OBS = len(SCAN_HISTORY) * spec.N_OBS_BEAMS + 3 + 3 + 2 + 1 + 2 + 2


def motion_lookahead(name):
    """Subgoal lookahead (m) the motion policy expects: its training setting, or LOOKAHEAD."""
    if name == 'downhill':
        return LOOKAHEAD
    from cash_rl.train import model_subgoal_lookahead
    lookahead = model_subgoal_lookahead(name)
    if not lookahead:
        raise ValueError(f'{name} was not trained with a subgoal lookahead')
    return lookahead


def motion_policy(name):
    """Return policy(base_env, obs) -> action for 'downhill' or a train.py model zip.

    Zips run through a numpy copy of the actor (fast enough inside every
    SubprocVecEnv worker, no torch per step).
    """
    if name == 'downhill':
        from cash_rl.scripted import downhill
        return lambda base, obs: downhill(base)
    from cash_rl.dagger import actor_action, actor_weights
    from cash_rl.train import load_model
    model = load_model(name)
    if getattr(model, 'observation_space', None) is None \
            or model.observation_space.shape != (spec.N_OBS,):
        raise ValueError(f'{name}: motion policy must take the current {spec.N_OBS}-dim obs')
    weights = actor_weights(model.policy)
    return lambda base, obs: actor_action(weights, obs)


class SubgoalObserver:
    """Builds the subgoal agent's observation from a HallwayRaycastEnv's state."""

    def __init__(self):
        self.scans = []

    def reset(self, base):
        self.scans = [base._ranges.copy()] * (max(SCAN_HISTORY) + 1)

    def push(self, base):
        """Record the latest scan; call once per decision."""
        self.scans = [base._ranges.copy()] + self.scans[:max(SCAN_HISTORY)]

    def observation(self, base, prev_action):
        """Return the N_OBS observation."""
        scans = np.concatenate([self.scans[i] for i in SCAN_HISTORY]) / spec.MAX_RANGE
        pose = base.pose
        route = base.route(max_length=LOOKAHEAD)
        center = route[-1] if len(route) > 1 else base.goal
        sub_distance, sub_bearing = spec.goal_in_robot_frame(pose, center)
        goal_distance, goal_bearing = spec.goal_in_robot_frame(pose, base.goal)
        room_left = room_right = 0.0
        lane = base.near_lane[2] if base.near_lane is not None else 0.0
        if len(route) > 5:
            direction = route[5] - route[0]
            direction /= max(np.linalg.norm(direction), 1e-9)
            room_left = base._lane_point(pose[:2], direction, 10.0)[1]
            room_right = -base._lane_point(pose[:2], direction, -10.0)[1]
        linear, angular = base._velocity
        return np.concatenate([
            scans,
            [min(sub_distance / LOOKAHEAD, 1.0), np.cos(sub_bearing), np.sin(sub_bearing)],
            [min(goal_distance / spec.MAX_GOAL_DISTANCE, 1.0), np.cos(goal_bearing),
             np.sin(goal_bearing)],
            [min(room_left / ROOM_SCALE, 1.0), min(room_right / ROOM_SCALE, 1.0)],
            [lane / MAX_LANE_OFFSET],
            [linear / spec.MAX_LINEAR_VEL, angular / spec.MAX_ANGULAR_VEL],
            np.clip(prev_action, -1.0, 1.0),
        ]).astype(np.float32)


def observation_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(N_OBS,), dtype=np.float32)


def apply_decision(base, action):
    """Set the lane from a subgoal-agent action; return True if it says wait."""
    base.lane_offset = float(np.clip(action[0], -1.0, 1.0)) * MAX_LANE_OFFSET
    return bool(action[1] > 0.0)


def task_reward(reward, info):
    """Return the step's reward without the motion agent's lane penalty."""
    error = info.get('lane_error', np.nan)
    return reward + (spec.subgoal_lane_penalty(error) if np.isfinite(error) else 0.0)


class SubgoalEnv(gym.Env):
    """One step = one subgoal decision, then DECISION_STEPS steps of the motion agent.

    env_kwargs go to HallwayRaycastEnv (subgoal_lookahead is the motion
    policy's, see motion_lookahead, and lanes are pinned: this agent sets
    them). info is the last control step's (without a NaN 'lane_error'), plus 'waiting'.
    """

    metadata = {'render_modes': []}

    def __init__(self, motion='downhill', **env_kwargs):
        super().__init__()
        env_kwargs = dict(env_kwargs, subgoal_lookahead=motion_lookahead(motion),
                          subgoal_offset=0.0)
        self.base = HallwayRaycastEnv(**env_kwargs)
        self.motion = motion_policy(motion)
        self.observer = SubgoalObserver()
        self.observation_space = observation_space()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.door_goals = self.base.door_goals
        self.spawn_pose = self.base.spawn_pose
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._motion_obs = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {}, lane_offset=0.0)
        self._motion_obs, info = self.base.reset(seed=seed, options=options)
        self.observer.reset(self.base)
        self._prev_action[:] = 0.0
        return self.observer.observation(self.base, self._prev_action), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        wait = apply_decision(self.base, action)
        self._motion_obs = self.base.observation()  # same scans, subgoal in the new lane
        total, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(DECISION_STEPS):
            command = np.zeros(2) if wait else self.motion(self.base, self._motion_obs)
            self._motion_obs, reward, terminated, truncated, info = self.base.step(command)
            total += task_reward(reward, info)
            if terminated or truncated:
                break
        self.observer.push(self.base)
        self._prev_action = action
        info['waiting'] = wait
        if not np.isfinite(info.get('lane_error', 0.0)):
            del info['lane_error']  # NaN (no lane charged) breaks info comparisons
        return (self.observer.observation(self.base, self._prev_action), total,
                terminated, truncated, info)


class SubgoalLayer(gym.Wrapper):
    """Runs a trained subgoal agent over a HallwayRaycastEnv driven by any motion policy.

    The wrapped env needs a subgoal lookahead (the motion policy's). Every
    DECISION_STEPS steps the agent sets the lane and may order a wait, during
    which the motion policy's actions are replaced by a stop. info adds
    'decision_offset' and 'waiting'.
    """

    def __init__(self, env, agent_path):
        super().__init__(env)
        if not env.unwrapped.subgoal_lookahead:
            raise ValueError('the subgoal agent needs a subgoal lookahead')
        from cash_rl.train import load_model
        self.agent = load_model(agent_path)
        self.observer = SubgoalObserver()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._wait = False
        self._steps = 0

    def _decide(self, info):
        base = self.env.unwrapped
        if self._steps:
            self.observer.push(base)
        obs = self.observer.observation(base, self._prev_action)
        action, _ = self.agent.predict(obs, deterministic=True)
        self._prev_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._wait = apply_decision(base, self._prev_action)
        info['decision_offset'] = base.lane_offset
        info['waiting'] = self._wait
        return base.observation(), info

    def reset(self, *, seed=None, options=None):
        options = dict(options or {}, lane_offset=0.0)
        _, info = self.env.reset(seed=seed, options=options)
        self.observer.reset(self.env.unwrapped)
        self._prev_action[:] = 0.0
        self._steps = 0
        return self._decide(info)

    def step(self, action):
        if self._wait:
            action = np.zeros(2, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        info['waiting'] = self._wait
        if not (terminated or truncated) and self._steps % DECISION_STEPS == 0:
            obs, info = self._decide(info)
        return obs, reward, terminated, truncated, info


def main():
    from cash_rl.train import algo_kwargs, GoalEvalCallback, runs_dir
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    parser = argparse.ArgumentParser(description='Train the subgoal agent with PPO.')
    parser.add_argument('--motion', required=True,
                        help='frozen motion policy: a train.py model zip trained with '
                             "--subgoal-lookahead (and lanes), or 'downhill'")
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='decisions (each runs DECISION_STEPS control steps)')
    parser.add_argument('--pedestrians', type=int, default=1)
    parser.add_argument('--encounter-prob', type=float, default=0.7,
                        help='share of training episodes whose first pedestrian starts on the '
                             "robot's route walking toward it (eval: always)")
    parser.add_argument('--goal-distance-range', type=float, nargs=2, default=(1.0, 25.0))
    parser.add_argument('--eval-random', type=int, default=40)
    parser.add_argument('--eval-freq', type=int, default=50_000)
    parser.add_argument('--n-envs', type=int, default=12)
    parser.add_argument('--map', help="map name in maps/ (e.g. 'cbu_floor2') or .yaml path")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', help='run name (default: subgoal_<timestamp>)')
    args = parser.parse_args()

    map_yaml = use_map(args.map) if args.map else os.path.abspath(default_map_yaml())
    motion = args.motion if args.motion == 'downhill' else os.path.abspath(args.motion)
    name = args.name or f'subgoal_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    run_dir = os.path.join(runs_dir(), name)
    os.makedirs(run_dir, exist_ok=False)
    kwargs = algo_kwargs('ppo', args.n_envs)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        # algo 'ppo' so train.load_model loads it; subgoal_agent marks what it is.
        json.dump({'args': dict(vars(args), algo='ppo', motion=motion,
                                motion_lookahead=motion_lookahead(motion)),
                   'map_yaml': map_yaml,
                   'subgoal_agent': True, 'n_envs': args.n_envs, 'algo_kwargs': kwargs,
                   'decision_steps': DECISION_STEPS,
                   'spec': {k: v for k, v in vars(spec).items()
                            if k.isupper() and isinstance(v, (int, float, str, dict, tuple))}},
                  f, indent=2, default=str)

    env_kwargs = {'motion': motion, 'goal_distance_range': tuple(args.goal_distance_range),
                  'n_pedestrians': args.pedestrians, 'encounter_prob': args.encounter_prob}
    train_env = make_vec_env(SubgoalEnv, n_envs=args.n_envs, seed=args.seed,
                             env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv,
                             monitor_kwargs={'info_keywords': ('is_success',)})
    # Scored on the encounter test: what this agent is for.
    eval_env = SubgoalEnv(motion=motion, n_pedestrians=args.pedestrians, encounter_prob=1.0)
    model = PPO('MlpPolicy', train_env, seed=args.seed, verbose=0, tensorboard_log=run_dir,
                **kwargs)
    callbacks = [
        GoalEvalCallback(eval_env, args.eval_random, args.eval_freq, run_dir),
        CheckpointCallback(max(250_000 // args.n_envs, 1), os.path.join(run_dir, 'checkpoints'),
                           name_prefix='subgoal'),
    ]
    print(f'Training the subgoal agent for {args.timesteps:,} decisions with {args.n_envs} '
          f'envs on {os.path.basename(map_yaml)}, motion {os.path.basename(motion)}, '
          f'{args.pedestrians} pedestrian(s) -> {run_dir}', flush=True)
    model.learn(args.timesteps, callback=callbacks)
    model.save(os.path.join(run_dir, 'final_model'))
    print(f'Done -> {run_dir}/final_model.zip')


if __name__ == '__main__':
    main()
