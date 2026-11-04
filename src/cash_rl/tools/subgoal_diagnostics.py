#!/usr/bin/env python3
r"""How a decision layer uses its lanes and waits: does it react to pedestrians?

Plays encounter episodes (1 pedestrian walking at the robot) with a motion policy under
the learned subgoal agent and/or the rule-based layer, and prints per layer: outcomes,
how far off center the lane is and how much it moves per step with no pedestrian
within NEAR m vs with one, the share of steps spent waiting, and how far the robot
was from its lane in the last 2 s before each pedestrian hit.

A reacting layer keeps the center lane with nobody near and moves when someone comes;
the first learned agents (2026-10-31) sat ~0.8 m off center all the time instead.

Run from ~/ros2_ws with the workspace sourced:
  python3 src/cash_rl/tools/subgoal_diagnostics.py runs/tegr_ppo_la15_s0/final_model.zip \
      --agent runs/tegr_subgoal_la15_s0/final_model.zip --rule
  python3 src/cash_rl/tools/subgoal_diagnostics.py downhill --agent runs/x/final_model.zip
"""
import argparse

import numpy as np

NEAR = 4.0  # m, pedestrian gap counted as "pedestrian near"


def run(env, base, motion, episodes, seed):
    from cash_rl.scripted import downhill
    out = {'success': 0, 'pedestrian': 0, 'wall': 0, 'timeout': 0}
    lane_abs = {False: [], True: []}
    lane_move = {False: [], True: []}
    waits = {False: [], True: []}
    pre_hit_error = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        previous, done, errors = base.lane_offset, False, []
        while not done:
            action = downhill(base) if motion == 'downhill' \
                else motion.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            near = info.get('min_pedestrian_gap', np.inf) < NEAR
            lane_abs[near].append(abs(base.lane_offset))
            lane_move[near].append(abs(base.lane_offset - previous))
            waits[near].append(bool(info.get('waiting', False)))
            previous = base.lane_offset
            error = info.get('lane_error', np.nan)
            if np.isfinite(error):
                errors.append(abs(error))
        if info.get('is_success'):
            out['success'] += 1
        elif info.get('hit_pedestrian'):
            out['pedestrian'] += 1
            pre_hit_error += errors[-20:]
        elif terminated:
            out['wall'] += 1
        else:
            out['timeout'] += 1

    def mean(xs):
        return f'{np.mean(xs):.3f}' if xs else '  -  '
    return (f"{out['success']:4d}  {out['pedestrian']:4d}  {out['wall']:4d}  {out['timeout']:4d}"
            f"   {mean(lane_abs[False])}  {mean(lane_abs[True])}"
            f"   {mean(lane_move[False])}  {mean(lane_move[True])}"
            f"   {mean(waits[False])}  {mean(waits[True])}   {mean(pre_hit_error)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('motion', help="motion policy: train.py model zip or 'downhill'")
    parser.add_argument('--agent', action='append', default=[], metavar='ZIP',
                        help='learned subgoal agent (repeatable)')
    parser.add_argument('--rule', action='store_true', help='also the rule-based layer')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10_000)
    parser.add_argument('--map', default='cbu_floor2')
    args = parser.parse_args()

    from cash_rl.occupancy_map import use_map
    use_map(args.map)
    from cash_rl.decision import DecisionLayer
    from cash_rl.raycast_env import HallwayRaycastEnv
    from cash_rl.subgoal_agent import motion_lookahead, SubgoalLayer
    from cash_rl.train import load_model
    motion = 'downhill' if args.motion == 'downhill' else load_model(args.motion)
    lookahead = motion_lookahead(args.motion)
    layers = [(path, lambda b, p=path: SubgoalLayer(b, p)) for path in args.agent]
    if args.rule:
        layers.append(('rule-based layer', DecisionLayer))
    if not layers:
        parser.error('give --agent ZIP and/or --rule')
    print(f'{args.episodes} encounter episodes, motion {args.motion}; '
          f'"near" = pedestrian within {NEAR} m')
    print(f'{"":>4}succ  ped  wall  t/o   |lane| far/near   move/step far/near'
          f'   wait far/near   lane err before ped hit')
    for label, make in layers:
        base = HallwayRaycastEnv(n_pedestrians=1, encounter_prob=1.0,
                                 subgoal_lookahead=lookahead)
        print(f'{label}\n    {run(make(base), base, motion, args.episodes, args.seed)}',
              flush=True)


if __name__ == '__main__':
    main()
