#!/usr/bin/env python3
r"""Find and replay the collisions of a motion agent following a pinned subgoal lane.

Runs the same held-out episodes as `evaluate --random N --lane-offset X` (same seeds, so the
collision count matches the eval file), prints one row per collision, then shows them:

  overview   one figure: map with every colliding episode's path (blue), the lane it was asked
             to follow at the robot (orange dashes) and the crash footprint (red)
  replay     each collision animated, zoomed on the robot, last --tail seconds before the crash

Run from ~/ros2_ws with the workspace sourced (or PYTHONPATH=src/cash_rl). GUI windows unless
--save DIR (writes overview.png and collision_<i>.gif there).

Examples:
  python3 src/cash_rl/tools/lane_collisions.py runs/tegr_ppo_la15_s0/final_model.zip
  python3 src/cash_rl/tools/lane_collisions.py runs/tegr_ppo_la15_s0/final_model.zip \
      --replay --tail 6 --speed 0.5
  python3 src/cash_rl/tools/lane_collisions.py runs/tegr_ppo_la15_s0/final_model.zip \
      --replay --save /tmp/right_lane
"""
import argparse
import os

from cash_rl import spec
from cash_rl.evaluate import HELD_OUT_SEED_BASE
from cash_rl.occupancy_map import use_map
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.train import load_model, model_lane_hold, model_map_yaml, model_subgoal_lookahead
from cash_rl.view_env import Scene, footprint_corners

import numpy as np


def play(env, model, seed, lane_offset, record=False):
    """Run one evaluate-style episode; return (info, frames or None)."""
    obs, _ = env.reset(seed=seed, options={'lane_offset': lane_offset})
    frames = [] if record else None
    trail, lanes = [env.pose[:2].copy()], []
    while True:
        lanes.append(env.robot_lane[0].copy() if env.robot_lane is not None
                     else np.full(2, np.nan))
        if record:
            frames.append({'pose': env.pose.copy(), 'ranges': env._ranges.copy(),
                           'subgoal': env.subgoal.copy(), 'goal': env.goal.copy(),
                           'trail': np.array(trail), 'lanes': np.array(lanes)})
        action = model.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, info = env.step(action)
        trail.append(env.pose[:2].copy())
        if terminated or truncated:
            info['trail'], info['lanes'] = np.array(trail), np.array(lanes)
            return info, frames


def describe(env):
    """Side of the nearest observed return at the crash, relative to the heading."""
    side = np.degrees(spec.OBS_BEAM_ANGLES[int(np.argmin(env._ranges))])
    where = 'front' if abs(side) < 45 else 'left' if 0 < side < 135 else \
        'right' if -135 < side < 0 else 'rear'
    return where, side


def overview(env, crashes, title, save):
    """All colliding paths on the map."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(11, 11))
    Scene(ax, env)
    for c in crashes:
        info = c['info']
        ax.plot(*info['trail'].T, color='tab:blue', lw=1, alpha=0.7)
        ax.plot(*info['lanes'].T, '--', color='darkorange', lw=1)
        ax.add_patch(Polygon(footprint_corners(info['pose']), fc='red', ec='darkred', alpha=0.8))
        ax.annotate(str(c['index']), info['pose'][:2], xytext=(4, 4),
                    textcoords='offset points', color='darkred', fontsize=8)
    ax.set_title(title, fontsize=11)
    if save:
        path = os.path.join(save, 'overview.png')
        fig.savefig(path, dpi=110)
        print(f'wrote {path}')
    return fig


def replay(env, frames, text, args, save_path):
    """Animate the last --tail s before the crash, zoomed on the robot."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    frames = frames[-int(args.tail / spec.CONTROL_PERIOD):]
    frames += [frames[-1]] * 15
    fig, ax = plt.subplots(figsize=(8, 8))
    scene = Scene(ax, env)
    lane_line, = ax.plot([], [], '--', color='darkorange', lw=1.5, label='lane at robot')
    ax.legend(loc='upper right', fontsize=8)
    poses = np.array([f['pose'][:2] for f in frames])
    center, half = poses.mean(axis=0), max(np.ptp(poses, axis=0).max() / 2 + 1.5, 3.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)

    def draw(i):
        f = frames[i]
        lane_line.set_data(*f['lanes'].T)
        crashed = ' COLLISION' if i >= len(frames) - 16 else ''
        return scene.draw(f['pose'], f['ranges'], f['goal'], f['trail'],
                          f'{text}{crashed}', subgoal=f['subgoal']) + (lane_line,)

    interval = 1000 * spec.CONTROL_PERIOD / args.speed
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval, blit=False,
                         repeat=not save_path)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=round(1000 / interval)), dpi=70)
        print(f'wrote {save_path}')
        plt.close(fig)
    else:
        plt.show()


def main():
    """Collect the collisions, print them, then plot / replay."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model', help='train.py model zip')
    parser.add_argument('--lane-offset', type=float, default=-0.5,
                        help='pinned lane, m left of center (default -0.5 = right)')
    parser.add_argument('--random', type=int, default=200, help='held-out episodes (as evaluate)')
    parser.add_argument('--replay', action='store_true', help='also animate each collision')
    parser.add_argument('--only', type=int, nargs='+', metavar='I',
                        help='replay only these collision numbers (from the table)')
    parser.add_argument('--tail', type=float, default=5.0, help='seconds before the crash')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed multiplier')
    parser.add_argument('--save', metavar='DIR', help='write PNG/GIFs instead of windows')
    args = parser.parse_args()

    if model_map_yaml(args.model):
        use_map(model_map_yaml(args.model))
    lookahead = model_subgoal_lookahead(args.model)
    if not lookahead:
        parser.error('model was trained without subgoals; lanes need a lookahead')
    env = HallwayRaycastEnv(subgoal_lookahead=lookahead, subgoal_offset=0.0,
                            lane_hold=model_lane_hold(args.model))
    model = load_model(args.model)
    if args.save:
        import matplotlib
        matplotlib.use('Agg')
        os.makedirs(args.save, exist_ok=True)

    crashes = []
    for i in range(args.random):
        seed = HELD_OUT_SEED_BASE + i
        info, _ = play(env, model, seed, args.lane_offset)
        if info['collided']:
            crashes.append({'index': len(crashes), 'episode': i, 'seed': seed, 'info': info,
                            'steps': len(info['lanes'])})
            where, side = describe(env)
            lane = info['lanes'][-1]
            off = (np.hypot(*(info['pose'][:2] - lane)) if np.all(np.isfinite(lane))
                   else np.nan)
            charged = np.isfinite(info['lanes'][:, 0])
            crashes[-1]['row'] = (
                f'{len(crashes) - 1:3d}  random_{i:<4d} step {len(info["lanes"]):4d}  '
                f'at ({info["pose"][0]:6.1f}, {info["pose"][1]:6.1f})  '
                f'nearest return {where:5s} ({side:+4.0f} deg)  '
                f'm from lane {off:4.2f}  lane-charged steps {charged.mean():4.0%}')
    n = args.random
    print(f'{len(crashes)}/{n} collisions ({len(crashes) / n:.1%}), lane {args.lane_offset:+g} m, '
          f'lookahead {lookahead:g} m')
    print('  #  episode     step   crash pose            what was closest            '
          '(nan = lane waived at crash: turn/junction/last m)')
    for c in crashes:
        print(c['row'])
    if not crashes:
        return

    import matplotlib.pyplot as plt
    name = args.model.rstrip('/').split('/')[-2]
    overview(env, crashes, f'{name}: {len(crashes)} collisions in {n} episodes, '
             f'lane {args.lane_offset:+g} m', args.save)
    if not args.save:
        plt.show()
    if args.replay:
        for c in crashes:
            if args.only and c['index'] not in args.only:
                continue
            _, frames = play(env, model, c['seed'], args.lane_offset, record=True)
            text = f'#{c["index"]} random_{c["episode"]}  lane {args.lane_offset:+g} m'
            path = (os.path.join(args.save, f'collision_{c["index"]}.gif') if args.save
                    else None)
            replay(env, frames, text, args, path)


if __name__ == '__main__':
    main()
