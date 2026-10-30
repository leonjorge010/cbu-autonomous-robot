"""Train a hallway goal-reaching policy on HallwayRaycastEnv with Stable-Baselines3.

    ros2 run cash_rl train                                  # PPO, 3M steps
    ros2 run cash_rl train --algo sac --timesteps 1000000
    ros2 run cash_rl train --pedestrians 4                  # moving pedestrians
    ros2 run cash_rl train --pedestrians 4 --yielding-pedestrians  # that step around it
    tensorboard --logdir ~/ros2_ws/runs                     # watch it

Writes runs/<name>/ in the workspace (gitignored): config.json, TensorBoard
logs, eval.csv, checkpoints/, best_model.zip and final_model.zip.

Every --eval-freq steps the deterministic policy is evaluated on a fixed set:
each spec.DOOR_GOALS entry from spec.SPAWN_POSE (what HallwayEnv evaluates in
Gazebo) plus --eval-random fixed-seed random spawn/goal pairs. Success,
collision and timeout rates and mean return go to TensorBoard (eval/*) and
eval.csv — those, not the exploration-noisy training reward, are the learning
curves for the report.
"""
import argparse
import csv
import datetime
import json
import os

from cash_rl import spec
from cash_rl.occupancy_map import default_map_yaml, use_map
from cash_rl.raycast_env import DEFAULT_GOAL_DISTANCE_RANGE, HallwayRaycastEnv
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ALGOS = {'ppo': PPO, 'sac': SAC, 'td3': TD3}
# Seeds for the random part of the eval set; far from training seeds.
EVAL_SEED_BASE = 1_000_000


def runs_dir():
    """Return <workspace>/runs (the workspace holding maps/)."""
    return os.path.join(os.path.dirname(os.path.dirname(default_map_yaml())), 'runs')


def load_model(path):
    """Load a train.py model zip; the algo is read from the run's config.json.

    Models trained before scan stacking (spec.SINGLE_SCAN_N_OBS observations) are
    wrapped to take current N_OBS observations, keeping only the newest scan.
    """
    path = os.path.expanduser(path)
    directory = os.path.dirname(os.path.abspath(path))
    for candidate in (directory, os.path.dirname(directory)):  # run dir or checkpoints/
        config = os.path.join(candidate, 'config.json')
        if os.path.isfile(config):
            with open(config) as f:
                model = ALGOS[json.load(f)['args']['algo']].load(path, device='cpu')
            if model.observation_space.shape == (spec.SINGLE_SCAN_N_OBS,):
                return SingleScanModel(model)
            return model
    raise FileNotFoundError(f'No config.json next to or above {path}')


def _model_config(path):
    """Return the config.json of the run a model zip belongs to, or {}."""
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    for candidate in (directory, os.path.dirname(directory)):  # run dir or checkpoints/
        config = os.path.join(candidate, 'config.json')
        if os.path.isfile(config):
            with open(config) as f:
                return json.load(f)
    return {}


def model_map_yaml(path):
    """Map a train.py/dagger.py model was trained on (its config.json), or None.

    None for runs from before maps were recorded, which all used the hallway.
    """
    return _model_config(path).get('map_yaml')


def model_subgoal_lookahead(path):
    """--subgoal-lookahead a train.py/dagger.py model was trained with, or None."""
    return _model_config(path).get('args', {}).get('subgoal_lookahead')


def model_subgoal_offset(path):
    """--subgoal-offset a train.py/dagger.py model was trained with, or 0."""
    return _model_config(path).get('args', {}).get('subgoal_offset') or 0.0


class SingleScanModel:
    """Wrap a pre-scan-stacking model so predict() accepts N_OBS observations."""

    def __init__(self, model):
        self.model = model

    def predict(self, observation, **kwargs):
        return self.model.predict(spec.single_scan_observation(observation), **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def algo_kwargs(algo, n_envs):
    net = {'net_arch': [256, 256]}
    if algo == 'ppo':
        # ~8k-step rollouts (several full episodes per env per update); a power of
        # two >= 256 per env keeps the rollout a multiple of batch_size.
        n_steps = max(256, 2 ** int(round(np.log2(8192 / n_envs))))
        return {'n_steps': n_steps, 'batch_size': 256, 'n_epochs': 10,
                'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_range': 0.2, 'ent_coef': 0.0, 'policy_kwargs': net}
    return {'learning_rate': 3e-4, 'buffer_size': 1_000_000, 'batch_size': 256,
            'gamma': 0.99, 'learning_starts': 10_000, 'policy_kwargs': net}


def evaluate(model, env, episodes):
    """Run (name, seed, options) episodes deterministically; return outcome dicts.

    Besides outcome and return: 'path' = distance driven (m), 'start_distance' =
    geodesic start distance (m), 'reverse' = share of steps commanding reverse,
    'override' = share of steps a safety.SafetyWrapper changed the action,
    'lane_errors' = |info['lane_error']| on steps charged a lane penalty (m, a list).
    """
    results = []
    for name, seed, options in episodes:
        obs, info = env.reset(seed=seed, options=options)
        start_distance, prev = info['goal_distance'], info['pose'][:2].copy()
        total, steps, path, reversing, overrides = 0.0, 0, 0.0, 0, 0
        lane_errors = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            reversing += action[0] < 0
            overrides += bool(info.get('safety_override', False))
            if np.isfinite(info.get('lane_error', np.nan)):
                lane_errors.append(abs(info['lane_error']))
            path += float(np.hypot(*(info['pose'][:2] - prev)))
            prev = info['pose'][:2].copy()
            if terminated or truncated:
                break
        results.append({'name': name, 'success': bool(info['is_success']),
                        'collision': bool(info['collided']),
                        'timeout': not terminated, 'return': total, 'steps': steps,
                        'path': path, 'start_distance': start_distance,
                        'reverse': reversing / steps, 'override': overrides / steps,
                        'lane_errors': lane_errors})
    return results


class GoalEvalCallback(BaseCallback):
    """Periodic deterministic eval on door goals + fixed random pairs; keeps the best."""

    def __init__(self, eval_env, n_random, eval_freq, run_dir):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.run_dir = run_dir
        # Door goals from the spawn on the hallway map; other maps have none.
        self.episodes = [(name, 0, {'spawn': eval_env.spawn_pose, 'goal': goal})
                         for name, goal in sorted(eval_env.door_goals.items())]
        self.episodes += [(f'random_{i}', EVAL_SEED_BASE + i, None) for i in range(n_random)]
        self.best = (-1.0, -np.inf)
        self.last_eval = 0
        self.csv_path = os.path.join(run_dir, 'eval.csv')
        self.fields = ['timesteps', 'success_rate', 'collision_rate', 'timeout_rate',
                       'mean_return', 'door_success_rate', 'random_success_rate',
                       'mean_steps_success']

    def _on_training_start(self):
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.fields)
        self._run_eval()

    def _on_step(self):
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self._run_eval()
        return True

    def _on_training_end(self):
        self._run_eval()

    def _run_eval(self):
        self.last_eval = self.num_timesteps
        results = evaluate(self.model, self.eval_env, self.episodes)

        def rate(key, subset=results):
            return float(np.mean([r[key] for r in subset]))

        doors = [r for r in results if not r['name'].startswith('random_')]
        randoms = [r for r in results if r['name'].startswith('random_')]
        succeeded = [r['steps'] for r in results if r['success']]
        row = {
            'timesteps': self.num_timesteps,
            'success_rate': rate('success'),
            'collision_rate': rate('collision'),
            'timeout_rate': rate('timeout'),
            'mean_return': rate('return'),
            'door_success_rate': rate('success', doors) if doors else float('nan'),
            'random_success_rate': rate('success', randoms) if randoms else float('nan'),
            'mean_steps_success': float(np.mean(succeeded)) if succeeded else float('nan'),
        }
        for key, value in row.items():
            if key != 'timesteps':
                self.logger.record(f'eval/{key}', value)
        for r in doors:
            self.logger.record(f'eval_doors/{r["name"]}', float(r['success']))
        self.logger.dump(self.num_timesteps)
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([row[k] for k in self.fields])

        score = (row['success_rate'], row['mean_return'])
        if score > self.best:
            self.best = score
            self.model.save(os.path.join(self.run_dir, 'best_model'))
        print(f'[eval @ {self.num_timesteps:>9,}] success {row["success_rate"]:.0%} '
              f'(doors {row["door_success_rate"]:.0%}) collision {row["collision_rate"]:.0%} '
              f'timeout {row["timeout_rate"]:.0%} return {row["mean_return"]:+.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--algo', choices=sorted(ALGOS), default='ppo')
    parser.add_argument('--timesteps', type=int, default=3_000_000)
    parser.add_argument('--n-envs', type=int, default=None,
                        help='parallel envs (default: 12 for ppo, 4 for sac/td3)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-freq', type=int, default=100_000, help='timesteps between evals')
    parser.add_argument('--eval-random', type=int, default=20,
                        help='fixed random spawn/goal pairs in each eval, besides the doors')
    parser.add_argument('--goal-distance-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        default=DEFAULT_GOAL_DISTANCE_RANGE,
                        help='geodesic distance range (m) of random training goals')
    parser.add_argument('--junction-prob', type=float, default=0.0,
                        help='share of training episodes that start and end on opposite '
                             'sides of the junction')
    parser.add_argument('--pedestrians', type=int, default=0,
                        help='moving pedestrians per episode, in training and eval')
    parser.add_argument('--yielding-pedestrians', action='store_true',
                        help='pedestrians slow down and step around the robot instead of '
                             'walking into it (how pedestrian runs before 2026-10-21 trained)')
    parser.add_argument('--encounter-prob', type=float, default=0.0,
                        help='share of training episodes whose first pedestrian starts on '
                             "the robot's route, walking toward it (eval keeps random starts)")
    parser.add_argument('--no-lane-penalty', action='store_true',
                        help='train without the pedestrian lane penalty (spec.PEDESTRIAN_LANE_*)')
    parser.add_argument('--subgoal-lookahead', type=float, default=None, metavar='M',
                        help='observe a subgoal M m ahead on the geodesic route instead of the '
                             'goal, in training and eval (reward still uses the goal)')
    parser.add_argument('--subgoal-offset', type=float, default=0.0, metavar='W',
                        help='put the subgoal in a random lane up to W m left/right of the '
                             'corridor center and pay for staying in it (needs '
                             '--subgoal-lookahead), in training and eval')
    parser.add_argument('--lr-schedule', choices=['constant', 'linear'], default='constant',
                        help='learning rate: constant, or linear decay to 0 over training')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='PPO entropy bonus (default 0.0, from algo_kwargs)')
    parser.add_argument('--map', help="map name in maps/ (e.g. 'cbu_floor2') or .yaml path "
                        '(default: $CASH_MAP_YAML, else hallway); door evals are hallway only')
    parser.add_argument('--checkpoint-freq', type=int, default=250_000)
    parser.add_argument('--name', help='run name (default: <algo>_<timestamp>)')
    args = parser.parse_args()
    if args.subgoal_offset and not args.subgoal_lookahead:
        parser.error('--subgoal-offset needs --subgoal-lookahead')

    # Set before any env exists, so SubprocVecEnv workers load the same map.
    map_yaml = use_map(args.map) if args.map else os.path.abspath(default_map_yaml())
    n_envs = args.n_envs or (12 if args.algo == 'ppo' else 4)
    name = args.name or f'{args.algo}_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    run_dir = os.path.join(runs_dir(), name)
    os.makedirs(run_dir, exist_ok=False)

    kwargs = algo_kwargs(args.algo, n_envs)
    if args.ent_coef is not None:
        if args.algo != 'ppo':
            parser.error('--ent-coef is PPO only')
        kwargs['ent_coef'] = args.ent_coef
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'args': vars(args), 'map_yaml': map_yaml, 'n_envs': n_envs,
                   'algo_kwargs': kwargs,
                   'spec': {k: v for k, v in vars(spec).items()
                            if k.isupper() and isinstance(v, (int, float, str, dict, tuple))}},
                  f, indent=2, default=str)

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env_kwargs = {'goal_distance_range': tuple(args.goal_distance_range),
                  'junction_prob': args.junction_prob,
                  'n_pedestrians': args.pedestrians,
                  'pedestrians_yield': args.yielding_pedestrians,
                  'lane_penalty': not args.no_lane_penalty,
                  'encounter_prob': args.encounter_prob,
                  'subgoal_lookahead': args.subgoal_lookahead,
                  'subgoal_offset': args.subgoal_offset}
    train_env = make_vec_env(HallwayRaycastEnv, n_envs=n_envs, seed=args.seed,
                             env_kwargs=env_kwargs,
                             vec_env_cls=vec_cls,
                             monitor_kwargs={'info_keywords': ('is_success',)})
    # Eval always uses the default goal range, so every run is scored on the same
    # random spawn/goal pairs regardless of its training goal range. It does use
    # the training pedestrians: runs are comparable only at equal counts and behavior.
    # Pedestrian starts stay random (no encounter_prob), for the same reason.
    eval_env = HallwayRaycastEnv(n_pedestrians=args.pedestrians,
                                 pedestrians_yield=args.yielding_pedestrians,
                                 lane_penalty=not args.no_lane_penalty,
                                 subgoal_lookahead=args.subgoal_lookahead,
                                 subgoal_offset=args.subgoal_offset)

    model_kwargs = dict(kwargs)
    if args.lr_schedule == 'linear':
        initial_lr = kwargs.get('learning_rate', 3e-4)
        # SB3 passes progress_remaining, which goes from 1 to 0 over training.
        model_kwargs['learning_rate'] = lambda progress: initial_lr * progress
    model = ALGOS[args.algo]('MlpPolicy', train_env, seed=args.seed, verbose=0,
                             tensorboard_log=run_dir, **model_kwargs)
    callbacks = [
        GoalEvalCallback(eval_env, args.eval_random, args.eval_freq, run_dir),
        CheckpointCallback(save_freq=max(1, args.checkpoint_freq // n_envs),
                           save_path=os.path.join(run_dir, 'checkpoints'), name_prefix='model'),
    ]
    print(f'Training {args.algo.upper()} for {args.timesteps:,} steps with {n_envs} envs '
          f'on {os.path.basename(map_yaml)} -> {run_dir}', flush=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name='tb',
                    progress_bar=False)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        train_env.close()
    print(f'Done. Best eval: success {callbacks[0].best[0]:.0%}, return '
          f'{callbacks[0].best[1]:+.1f} -> {run_dir}/best_model.zip')


if __name__ == '__main__':
    main()
