r"""Teacher-student training (DAgger): imitate scripted.avoider from the policy's own observation.

    ros2 run cash_rl dagger --pedestrians 1 --encounter-prob 0.7 --name dagger_p1
    ros2 run cash_rl evaluate runs/dagger_p1/final_model.zip avoider --random 200 --pedestrians 1 \
        --encounter-prob 1

The teacher, scripted.avoider, cheats: it reads the map's geodesic field and the
pedestrians' true positions and velocities. The student is the same MLP actor
train.py's PPO uses and sees only the normal observation (stacked scans, goal,
previous action). Round 0 is behavior cloning: the teacher drives, each step's
(observation, teacher action) is recorded. Each later round the student drives,
the teacher's action is executed instead with probability beta (halved every
round), and the teacher labels every visited state with what it would have
done. The labels are added to one growing dataset and the student is refit on
all of it, so it learns to recover from states only its own mistakes lead to.
No reward is used.

Writes runs/<name>/ like train.py: config.json (algo 'ppo', so load_model,
evaluate, view_env and demo work unchanged), eval.csv (one row per round),
checkpoints/round_NN.zip, best_model.zip and final_model.zip. Each round's
student is scored deterministically on --eval-episodes encounter episodes
(encounter_prob 1, seeds of train.py's periodic eval, disjoint from evaluate's
held-out seeds).
"""
import argparse
import csv
import datetime
import json
import multiprocessing
import os
import time

from cash_rl import scripted, spec
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.train import algo_kwargs, EVAL_SEED_BASE, evaluate, runs_dir
import numpy as np

# Collection seeds: (seed * 1000 + round) * 100 + worker, below EVAL_SEED_BASE.
_worker_env = None


def actor_weights(policy):
    """Numpy copies of an SB3 MlpPolicy's actor layers, as [(W, b), ...]."""
    layers = [m for m in policy.mlp_extractor.policy_net if hasattr(m, 'weight')]
    layers.append(policy.action_net)
    return [(m.weight.detach().cpu().numpy().astype(np.float64),
             m.bias.detach().cpu().numpy().astype(np.float64)) for m in layers]


def actor_action(weights, observation):
    """Deterministic action of an SB3 PPO MlpPolicy (tanh hidden layers) in numpy."""
    h = np.asarray(observation, dtype=np.float64)
    for w, b in weights[:-1]:
        h = np.tanh(w @ h + b)
    w, b = weights[-1]
    return np.clip(w @ h + b, -1.0, 1.0)


def _init_worker(env_kwargs):
    global _worker_env
    _worker_env = HallwayRaycastEnv(**env_kwargs)


def collect(task):
    """Run (seed, n_steps, beta, weights) in this worker's env; return labelled steps.

    weights None means the teacher drives (beta is ignored). Returns observations,
    teacher actions, and (success, collision) per finished episode.
    """
    seed, n_steps, beta, weights = task
    env = _worker_env
    base = env.unwrapped
    rng = np.random.default_rng(seed)
    observations = np.empty((n_steps, spec.N_OBS), dtype=np.float32)
    labels = np.empty((n_steps, 2), dtype=np.float32)
    outcomes = []
    obs, _ = env.reset(seed=seed)
    for i in range(n_steps):
        teacher = scripted.avoider(base)
        observations[i], labels[i] = obs, teacher
        if weights is None or rng.random() < beta:
            action = teacher
        else:
            action = actor_action(weights, obs)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcomes.append((bool(info['is_success']), bool(info['collided'])))
            obs, _ = env.reset()
    return observations, labels, outcomes


def fit(policy, optimizer, observations, labels, epochs, batch_size, rng):
    """Regress the actor's mean action onto the teacher's; return the last epoch's MSE."""
    import torch
    obs_t = torch.as_tensor(observations)
    label_t = torch.as_tensor(labels)
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(len(observations)))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            features = policy.extract_features(obs_t[idx])
            latent_pi = policy.mlp_extractor.forward_actor(features)
            loss = torch.mean((policy.action_net(latent_pi) - label_t[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
    policy.set_training_mode(False)
    return total / len(observations)


class NumpyActor:
    """predict() adapter so train.evaluate can score a weights snapshot."""

    def __init__(self, weights):
        self.weights = weights

    def predict(self, observation, **kwargs):
        return actor_action(self.weights, observation), None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pedestrians', type=int, default=1,
                        help='moving pedestrians per episode (collection and eval)')
    parser.add_argument('--encounter-prob', type=float, default=0.7,
                        help='share of collection episodes whose first pedestrian starts on '
                             "the robot's route, walking toward it")
    parser.add_argument('--rounds', type=int, default=10,
                        help='DAgger rounds after the behavior-cloning round 0')
    parser.add_argument('--steps-per-round', type=int, default=60_000,
                        help='labelled steps collected per round')
    parser.add_argument('--beta-decay', type=float, default=0.5,
                        help="round r executes the teacher's action with probability "
                             'beta_decay ** r (round 0: always)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='passes over the whole aggregated dataset per round')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--log-std', type=float, default=-1.0,
                        help='action log std stored in the saved model (only matters for '
                             'PPO fine-tuning later; -1 = std 0.37, as with ent_coef 0.001)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='encounter episodes to score the student on after each round')
    parser.add_argument('--workers', type=int, default=None,
                        help='collection processes (default: CPU count - 1)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', help='run name (default: dagger_<timestamp>)')
    args = parser.parse_args()

    import torch
    from stable_baselines3 import PPO

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    name = args.name or f'dagger_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    run_dir = os.path.join(runs_dir(), name)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=False)
    env_kwargs = {'n_pedestrians': args.pedestrians, 'encounter_prob': args.encounter_prob}
    # load_model reads args.algo; the student is an ordinary PPO MlpPolicy.
    config_args = dict(vars(args), algo='ppo')
    kwargs = algo_kwargs('ppo', 12)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'args': config_args, 'teacher': 'scripted.avoider', 'algo_kwargs': kwargs,
                   'spec': {k: v for k, v in vars(spec).items()
                            if k.isupper() and isinstance(v, (int, float, str, dict, tuple))}},
                  f, indent=2, default=str)

    model = PPO('MlpPolicy', HallwayRaycastEnv(**env_kwargs), seed=args.seed, verbose=0,
                device='cpu', **kwargs)
    policy = model.policy
    with torch.no_grad():
        policy.log_std.fill_(args.log_std)
    actor_params = list(policy.mlp_extractor.policy_net.parameters()) \
        + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    eval_env = HallwayRaycastEnv(n_pedestrians=args.pedestrians, encounter_prob=1.0)
    eval_episodes = [(f'encounter_{i}', EVAL_SEED_BASE + i, None)
                     for i in range(args.eval_episodes)]
    fields = ['round', 'beta', 'labels', 'mse', 'rollout_success', 'rollout_collision',
              'success_rate', 'collision_rate', 'timeout_rate', 'reverse', 'minutes']
    csv_path = os.path.join(run_dir, 'eval.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(fields)

    print(f'DAgger: {args.rounds + 1} rounds x {args.steps_per_round:,} labelled steps, '
          f'{workers} workers, {args.pedestrians} pedestrian(s), encounters '
          f'{args.encounter_prob:g} -> {run_dir}', flush=True)
    observations, labels = [], []
    best = (-1.0, 0.0)
    start = time.time()
    context = multiprocessing.get_context('spawn')
    with context.Pool(workers, initializer=_init_worker, initargs=(env_kwargs,)) as pool:
        for round_ in range(args.rounds + 1):
            beta = args.beta_decay ** round_
            weights = None if round_ == 0 else actor_weights(policy)
            chunk = int(np.ceil(args.steps_per_round / workers))
            tasks = [((args.seed * 1000 + round_) * 100 + w, chunk, beta, weights)
                     for w in range(workers)]
            outcomes = []
            for obs, lab, out in pool.map(collect, tasks):
                observations.append(obs)
                labels.append(lab)
                outcomes += out
            all_obs, all_labels = np.concatenate(observations), np.concatenate(labels)
            mse = fit(policy, optimizer, all_obs, all_labels, args.epochs, args.batch_size, rng)

            results = evaluate(NumpyActor(actor_weights(policy)), eval_env, eval_episodes)
            row = {
                'round': round_, 'beta': beta, 'labels': len(all_obs), 'mse': mse,
                'rollout_success': np.mean([s for s, _ in outcomes]) if outcomes else np.nan,
                'rollout_collision': np.mean([c for _, c in outcomes]) if outcomes else np.nan,
                'success_rate': np.mean([r['success'] for r in results]),
                'collision_rate': np.mean([r['collision'] for r in results]),
                'timeout_rate': np.mean([r['timeout'] for r in results]),
                'reverse': np.mean([r['reverse'] for r in results]),
                'minutes': (time.time() - start) / 60,
            }
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([row[k] for k in fields])
            model.save(os.path.join(run_dir, 'checkpoints', f'round_{round_:02d}'))
            score = (row['success_rate'], -row['collision_rate'])
            if score > best:
                best = score
                model.save(os.path.join(run_dir, 'best_model'))
            print(f'[round {round_:2d}] beta {beta:.3f} labels {len(all_obs):>9,} mse {mse:.4f} | '
                  f'rollouts {row["rollout_success"]:.0%} ok {row["rollout_collision"]:.0%} hit | '
                  f'encounter eval success {row["success_rate"]:.0%} collision '
                  f'{row["collision_rate"]:.0%} timeout {row["timeout_rate"]:.0%} reverse '
                  f'{row["reverse"]:.1%} | {row["minutes"]:.1f} min', flush=True)
    model.save(os.path.join(run_dir, 'final_model'))
    print(f'Done -> {run_dir}/final_model.zip (best round eval success {best[0]:.0%})')


if __name__ == '__main__':
    main()
