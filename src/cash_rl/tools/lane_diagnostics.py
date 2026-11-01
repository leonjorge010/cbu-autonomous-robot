#!/usr/bin/env python3
r"""Why doesn't a PPO motion agent follow subgoal lanes? Diagnostics behind WORKLOG 2026-10-31.

Run from ~/ros2_ws with the workspace sourced (or PYTHONPATH=src/cash_rl). MODEL is a train.py
zip path or 'downhill' where noted. All on HallwayRaycastEnv, no pedestrians.

  pinned MODEL...      lane pinned at +0.5 / 0 / -0.5 m: share of steps charged a lane penalty,
                       where the robot drives (m from center) and the lane penalty per episode
  sensitivity ZIP...   change in the policy's steering when the subgoal's bearing moves 9.5 deg
                       (0.5 m sideways at 3 m), and its action log_std
  tug ZIP...           steering in straight stretches vs the robot's sideways position (rows)
                       and the lane (columns): which pulls harder, LiDAR centering or the subgoal
  rewards ZIP          returns under training-style random lanes, split into lane penalty and
                       the rest, for downhill following / ignoring lanes, steering straight at
                       the observed subgoal, and ZIP
  pursuit              steering straight at the observed subgoal, for lookahead 3 / 2 / 1.5 m
                       and lane holds 2-6 s / 8-15 s: how well can the subgoal alone be followed

Examples:
  python3 src/cash_rl/tools/lane_diagnostics.py pinned \
      runs/tegr_ppo_lane_w2_s0/final_model.zip downhill
  python3 src/cash_rl/tools/lane_diagnostics.py tug runs/tegr_ppo_lane_w2_s0/final_model.zip
"""
import argparse

from cash_rl import scripted, spec
from cash_rl.occupancy_map import use_map
from cash_rl.raycast_env import HallwayRaycastEnv, LANE_HOLD_RANGE
from cash_rl.train import load_model

import numpy as np

GOAL = 3 * spec.N_OBS_BEAMS  # observation index of the goal distance; cos, sin follow
PINNED_LANES = (0.5, 0.0, -0.5)


def lane_env(lookahead=3.0, lane_hold=LANE_HOLD_RANGE):
    """Return a HallwayRaycastEnv with random lanes up to +-0.6 m."""
    return HallwayRaycastEnv(subgoal_lookahead=lookahead, subgoal_offset=0.6,
                             lane_hold=lane_hold)


def label(path):
    """Short name: the run directory, or 'downhill'."""
    return path if path == 'downhill' else path.rstrip('/').split('/')[-2]


def predictor(path, env):
    """Return obs -> action for a model zip or 'downhill'."""
    if path == 'downhill':
        return lambda obs: scripted.downhill(env)
    model = load_model(path)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def pursuit(obs):
    """Steer straight at the observed subgoal (no map, no LiDAR)."""
    bearing = np.arctan2(obs[GOAL + 2], obs[GOAL + 1])
    return np.array([1.0 if abs(bearing) < 0.4 else 0.1, np.clip(2 * bearing, -1, 1)])


def run(env, act, episodes, seed, options=None):
    """Play episodes; return per-step lists and per-episode outcomes."""
    out = {'charged': 0, 'steps': 0, 'applied': [], 'error': [], 'penalty': 0.0,
           'return': 0.0, 'success': 0, 'collision': 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep, options=options)
        done = False
        while not done:
            lane = env.robot_lane
            obs, reward, terminated, truncated, info = env.step(act(obs))
            done = terminated or truncated
            out['steps'] += 1
            out['return'] += reward
            error = info.get('lane_error', np.nan)
            if np.isfinite(error):
                out['charged'] += 1
                out['error'].append(error)
                out['penalty'] += spec.subgoal_lane_penalty(error)
                out['applied'].append(lane[2] if lane is not None else np.nan)
        out['success'] += info['is_success']
        out['collision'] += info['collided']
    return out


def pinned(args):
    """Lane pinned left / center / right: where the robot drives."""
    env = lane_env()
    for path in args.models:
        act = predictor(path, env)
        for offset in PINNED_LANES:
            r = run(env, act, args.episodes, 10_000, {'lane_offset': offset})
            error, applied = np.array(r['error']), np.array(r['applied'])
            position = applied + error
            charged = r['charged'] / r['steps']
            print(f'{label(path):24s} lane {offset:+.1f}: charged {charged:4.0%}  '
                  f'robot from center {np.nanmean(position):+.2f} m '
                  f'(sd {np.nanstd(position):.2f})  '
                  f'|err| {abs(error).mean():.2f}  '
                  f'lane penalty/ep {r["penalty"] / args.episodes:5.1f}',
                  flush=True)


def sensitivity(args):
    """Steering change per 9.5 deg of subgoal bearing."""
    import torch
    env = lane_env()
    for path in args.models:
        model = load_model(path)
        policy = model.policy
        states = []
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=500 + ep)
            done = False
            while not done:
                if env.robot_lane is not None:
                    states.append(obs.copy())
                action = model.predict(obs, deterministic=True)[0]
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        if not states:
            print(f'{label(path):24s} no lane-charged states; use more --episodes')
            continue
        states = np.array(states)
        bearing = np.arctan2(states[:, GOAL + 2], states[:, GOAL + 1])
        steer = {}
        for shift in (-9.5, 9.5):
            shifted = states.copy()
            shifted[:, GOAL + 1] = np.cos(bearing + np.radians(shift))
            shifted[:, GOAL + 2] = np.sin(bearing + np.radians(shift))
            with torch.no_grad():
                dist = policy.get_distribution(policy.obs_to_tensor(shifted)[0])
            steer[shift] = np.clip(dist.distribution.mean.numpy()[:, 1], -1, 1)
        change = (steer[9.5] - steer[-9.5]).mean() / 2 * spec.MAX_ANGULAR_VEL
        print(f'{label(path):24s} {len(states)} states: '
              f'steering {change:+.2f} rad/s per +9.5 deg '
              f'of subgoal bearing; log_std {policy.log_std.detach().numpy().round(2)}',
              flush=True)


def straight_spots(env, n=25):
    """(corridor center, direction, goal) at n spawns in straight, bounded stretches."""
    spots, seed = [], 0
    while len(spots) < n:
        seed += 1
        _, info = env.reset(seed=seed, options={'lane_offset': 0.0})
        if env.robot_lane is None or info['goal_distance'] < 6:
            continue
        route = env.route()
        (point, direction, _), _ = env._lane_at(route, 0, ahead=5)
        (_, ahead, _), bounded = env._lane_at(route, 40, ahead=5)
        if bounded and direction @ ahead >= 0.97:
            spots.append((point.copy(), direction.copy(), env.goal.copy()))
    return spots


def tug(args):
    """Steering vs robot position (LiDAR) and lane (subgoal)."""
    env = lane_env()
    spots = straight_spots(env)
    positions = (-0.5, -0.25, 0.0, 0.25, 0.5)
    for path in args.models:
        act = predictor(path, env)
        print(f'\n{label(path)}: mean steering (rad/s, + = left); rows robot m left of center, '
              'columns lane')
        print('   y \\ lane' + ''.join(f'{lane:+8.2f}' for lane in PINNED_LANES[::-1]))
        for y in positions:
            row = []
            for lane in PINNED_LANES[::-1]:
                steer = []
                for point, direction, goal in spots:
                    xy = point + y * np.array([-direction[1], direction[0]])
                    obs, _ = env.reset(seed=1, options={
                        'spawn': (*xy, np.arctan2(direction[1], direction[0])),
                        'goal': tuple(goal), 'lane_offset': lane})
                    steer.append(np.clip(act(obs)[1], -1, 1) * spec.MAX_ANGULAR_VEL)
                row.append(np.mean(steer))
            print(f'  {y:+.2f}     ' + ''.join(f'{v:+8.2f}' for v in row), flush=True)


def rewards(args):
    """Split returns into lane penalty and the rest."""
    env = lane_env()
    controllers = {'downhill follows lanes': lambda obs: scripted.downhill(env),
                   'downhill ignores lanes': lambda obs: scripted.downhill(env, lane_weight=0.0),
                   'steer at subgoal': pursuit,
                   label(args.model): predictor(args.model, env)}
    n = args.episodes
    for name, act in controllers.items():
        r = run(env, act, n, 20_000)
        print(f'{name:24s} success {r["success"] / n:4.0%} collision {r["collision"] / n:4.0%}  '
              f'|err| {np.mean(np.abs(r["error"])):.2f}  return/ep {r["return"] / n:6.1f} = '
              f'{-r["penalty"] / n:6.1f} lane + {(r["return"] + r["penalty"]) / n:6.1f} rest',
              flush=True)


def pursuit_sweep(args):
    """Steering at the subgoal across lookaheads and lane holds."""
    n = args.episodes
    for hold in (LANE_HOLD_RANGE, (8.0, 15.0)):
        for lookahead in (3.0, 2.0, 1.5):
            r = run(lane_env(lookahead, hold), pursuit, n, 20_000)
            print(f'hold {hold[0]:g}-{hold[1]:g} s, lookahead {lookahead:g} m: '
                  f'success {r["success"] / n:4.0%} collision {r["collision"] / n:4.0%}  '
                  f'|err| {np.mean(np.abs(r["error"])):.2f}  '
                  f'lane penalty/ep {r["penalty"] / n:5.1f}',
                  flush=True)


def main():
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--map', default='cbu_floor2', help='map name in maps/ or .yaml path')
    parser.add_argument('--episodes', type=int, default=None,
                        help='episodes per setting (defaults: pinned 40, sensitivity 15, '
                             'rewards 80, pursuit 60)')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, nargs in (('pinned', '+'), ('sensitivity', '+'), ('tug', '+')):
        sub.add_parser(name).add_argument('models', nargs=nargs)
    sub.add_parser('rewards').add_argument('model')
    sub.add_parser('pursuit')
    args = parser.parse_args()
    use_map(args.map)
    default_episodes = {'pinned': 40, 'sensitivity': 15, 'rewards': 80, 'pursuit': 60}
    if args.episodes is None:
        args.episodes = default_episodes.get(args.command, 0)
    {'pinned': pinned, 'sensitivity': sensitivity, 'tug': tug, 'rewards': rewards,
     'pursuit': pursuit_sweep}[args.command](args)


if __name__ == '__main__':
    main()
