#!/usr/bin/env python3
"""Sort the passing rule's crashes on evaluate's held-out goals.

Same seeds as ``evaluate --random N``. One label per collision, from the
decision that was in force before the contact:

  wall            the footprint hit the map, not a person
  behind          the person who hit was already behind the rule's window
  outside         they were ahead of 6 m or more than 1.5 m off the line
  no_room         both sides of that person are narrower than a pass
  wrong_side      the held side has no room and the other side does
  lane_clear      the lane the rule would drive did not overlap them
  known           the rule was already braking for that person and still hit

Prints a count per label. Run from the workspace with PYTHONPATH=src/cash_rl.
"""
import argparse
from collections import Counter

import numpy as np

from cash_rl.decision import (DecisionLayer, LATERAL_WINDOW, MAX_AHEAD, PASS_CLEARANCE,
                              RELEASE_BEHIND)
from cash_rl.evaluate import HELD_OUT_SEED_BASE
from cash_rl.occupancy_map import use_map
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.scripted import downhill


def _frame(layer):
    base = layer.env.unwrapped
    ahead, direction, normal = layer._route_frame(base)
    return base, ahead, direction, normal


def _room(base, index, direction, normal, side):
    lane, _, _ = base._lane_point(base.pedestrians[index], direction, side * 10.0)
    return side * float((lane - base.pedestrians[index]) @ normal)


def label_collision(layer, pre):
    """Name the cause of the collision that just happened.

    Geometry is the decision from before the contact. The person is whoever
    overlaps the footprint now. Room is measured at that person's position
    before the step, which is where the rule chose a side.
    """
    base = layer.env.unwrapped
    gaps = base.pedestrian_gaps()
    if len(gaps) == 0 or float(np.min(gaps)) >= 0.0 or pre['ahead'] is None:
        return 'wall'
    index = int(np.argmin(gaps))
    if index >= len(pre['ahead']):
        return 'outside'
    along = float(pre['ahead'][index])
    lateral = float(pre['lateral'][index])
    if along < -RELEASE_BEHIND:
        return 'behind'
    if along > MAX_AHEAD or abs(lateral) > LATERAL_WINDOW:
        return 'outside'
    if pre['blocking'] is None:
        return 'lane_clear'
    direction, normal = pre['direction'], pre['normal']
    saved = base.pedestrians
    base.pedestrians = pre['pedestrians']
    try:
        side = pre['sides'].get(index)
        if side is None:
            side = layer.decider._side_with_room(
                base, base.pedestrians[index], direction, normal)
        rooms = {s: _room(base, index, direction, normal, s) for s in (1.0, -1.0)}
    finally:
        base.pedestrians = saved
    if rooms[1.0] < PASS_CLEARANCE and rooms[-1.0] < PASS_CLEARANCE:
        return 'no_room'
    if rooms[side] < PASS_CLEARANCE:
        return 'wrong_side'
    return 'known'


def _snapshot(env):
    base, ahead, direction, normal = _frame(env)
    heading = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
    left = np.array([-heading[1], heading[0]])
    rel = base.pedestrians - base.pose[:2] if len(base.pedestrians) else np.zeros((0, 2))
    lane_gap = None
    if base.near_lane is not None and len(base.pedestrians):
        point, lane_dir, applied = base.near_lane
        lane_normal = np.array([-lane_dir[1], lane_dir[0]])
        center = point - applied * lane_normal
        lane_gap = np.abs((base.pedestrians - center) @ lane_normal - applied)
    return {'blocking': env._blocking_index(), 'sides': dict(env.decider.sides),
            'ahead': None if len(ahead) == 0 else np.array(ahead, copy=True),
            'lateral': None if len(ahead) == 0 else (rel @ normal).copy(),
            'direction': direction.copy(), 'normal': normal.copy(),
            'pedestrians': np.array(base.pedestrians, copy=True) if len(base.pedestrians) else None,
            'gaps': None if len(rel) == 0 else base.pedestrian_gaps().copy(),
            'body_ahead': None if len(rel) == 0 else (rel @ heading).copy(),
            'body_lat': None if len(rel) == 0 else (rel @ left).copy(),
            'lane_gap': None if lane_gap is None else lane_gap.copy(),
            'lane_error': base._robot_lane_error()}


def run(env, n, detail=False):
    counts = Counter()
    outcomes = Counter()
    details = []
    for i in range(n):
        obs, _ = env.reset(seed=HELD_OUT_SEED_BASE + i)
        while True:
            pre = _snapshot(env)
            action = downhill(env)
            obs, _, terminated, truncated, info = env.step(action)
            if not (terminated or truncated):
                continue
            if info['collided']:
                cause = label_collision(env, pre)
                counts[cause] += 1
                outcomes['collision'] += 1
                if detail and cause == 'lane_clear' and pre['gaps'] is not None:
                    index = int(np.argmin(env.env.unwrapped.pedestrian_gaps()))
                    details.append((float(pre['gaps'][index]), float(pre['body_ahead'][index]),
                                    abs(float(pre['body_lat'][index])),
                                    float(pre['lane_gap'][index]) if pre['lane_gap'] is not None else np.nan,
                                    abs(float(pre['lane_error'])), float(pre['ahead'][index])))
            elif info['is_success']:
                outcomes['success'] += 1
            else:
                outcomes['timeout'] += 1
            break
        del obs
    return counts, outcomes, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='cbu_floor2')
    parser.add_argument('--random', type=int, default=200)
    parser.add_argument('--pedestrians', type=int, default=1)
    parser.add_argument('--encounter-prob', type=float, default=1.0)
    parser.add_argument('--yield-prob', type=float, default=0.0)
    parser.add_argument('--cross-prob', type=float, default=0.0)
    parser.add_argument('--detail', action='store_true')
    args = parser.parse_args()
    use_map(args.map)
    env = DecisionLayer(HallwayRaycastEnv(
        n_pedestrians=args.pedestrians, encounter_prob=args.encounter_prob,
        yield_prob=args.yield_prob, cross_prob=args.cross_prob,
        subgoal_lookahead=1.5, subgoal_offset=0.0), wait=True)
    counts, outcomes, details = run(env, args.random, detail=args.detail)
    n = args.random
    print(f'pedestrians {args.pedestrians} yield {args.yield_prob:g} '
          f'cross {args.cross_prob:g}  n={n}', flush=True)
    print(f'  success {outcomes["success"] / n:.1%}  '
          f'collision {outcomes["collision"] / n:.1%}  '
          f'timeout {outcomes["timeout"] / n:.1%}', flush=True)
    for name, count in counts.most_common():
        print(f'  {name:12} {count:4}  {count / max(outcomes["collision"], 1):.0%} of crashes',
              flush=True)
    if details:
        cols = np.array(details)
        names = ('gap', 'body_ahead', 'body_lat', 'lane_gap', 'lane_err', 'route_ahead')
        print('  lane_clear medians: ' + '  '.join(
            f'{name} {np.nanmedian(cols[:, i]):.2f}' for i, name in enumerate(names)), flush=True)
        print('  lane_clear rows: gap body_ahead body_lat lane_gap lane_err route_ahead',
              flush=True)
        for row in cols:
            print('   ' + ' '.join(f'{v:6.2f}' for v in row), flush=True)


if __name__ == '__main__':
    main()
