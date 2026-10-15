"""Evaluate trained policies on the door goals (+ held-out random goals), 2D or Gazebo.

    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip              # 2D, doors
    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip --env gazebo
    ros2 run cash_rl evaluate runs/a/best_model.zip runs/b/best_model.zip --random 200

--env gazebo needs `ros2 launch cash_rl train_env.launch.py` running, and gives
the most faithful result when nothing else is loading the CPU: HallwayEnv steps
on wall-clock time, so a slow simulation changes the sim time per step.
Door episodes start at spec.SPAWN_POSE. --random N adds N random spawn/goal
pairs (2D only) on seeds held out from both training and train.py's periodic
eval, which picks best_model.zip and so would flatter it. Several models are
scored on identical episodes and summarized side by side.
"""
import argparse
import os

from cash_rl import spec
from cash_rl.train import EVAL_SEED_BASE, evaluate, load_model
import numpy as np

# Held-out random pairs: past train.py's periodic-eval seeds (EVAL_SEED_BASE + i).
HELD_OUT_SEED_BASE = EVAL_SEED_BASE + 100_000


def summarize(results):
    def mean(key, subset):
        return float(np.mean([r[key] for r in subset])) if subset else float('nan')
    ok = [r for r in results if r['success']]
    return {
        'success': mean('success', results),
        'collision': mean('collision', results),
        'timeout': mean('timeout', results),
        'time_s': mean('steps', ok) * spec.CONTROL_PERIOD,
        'path_ratio': float(np.mean([r['path'] / r['start_distance'] for r in ok]))
        if ok else float('nan'),
        'reverse': mean('reverse', results),
        'override': mean('override', results),
        'return': mean('return', results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('models', nargs='+', help='train.py model .zip files')
    parser.add_argument('--env', choices=['2d', 'gazebo'], default='2d')
    parser.add_argument('--repeats', type=int, default=1, help='episodes per door')
    parser.add_argument('--random', type=int, default=0,
                        help='held-out random spawn/goal pairs to add (2D only)')
    parser.add_argument('--safety', action='store_true',
                        help='run the policy behind the LiDAR safety layer (cash_rl.safety)')
    args = parser.parse_args()
    if args.random and args.env == 'gazebo':
        parser.error('--random is 2D only (HallwayEnv spawns from fixed poses)')

    if args.env == 'gazebo':
        from cash_rl.hallway_env import HallwayEnv
        env = HallwayEnv()
    else:
        from cash_rl.raycast_env import HallwayRaycastEnv
        env = HallwayRaycastEnv()
    if args.safety:
        from cash_rl.safety import SafetyWrapper
        env = SafetyWrapper(env)
    mode = f'{args.env}, safety layer' if args.safety else args.env
    doors = [(name, repeat, {'spawn': spec.SPAWN_POSE, 'goal': goal})
             for name, goal in sorted(spec.DOOR_GOALS.items())
             for repeat in range(args.repeats)]
    randoms = [(f'random_{i}', HELD_OUT_SEED_BASE + i, None) for i in range(args.random)]

    summaries = []
    try:
        for path in args.models:
            model = load_model(path)
            label = os.path.relpath(os.path.expanduser(path))
            print(f'\n== {label} ({mode})', flush=True)
            print(f'{"door":10} {"outcome":10} {"steps":>6} {"return":>8}', flush=True)
            door_results = []
            for episode in doors:
                result = evaluate(model, env, [episode])[0]
                door_results.append(result)
                outcome = ('reached' if result['success'] else
                           'collision' if result['collision'] else 'timeout')
                print(f'{result["name"]:10} {outcome:10} {result["steps"]:>6} '
                      f'{result["return"]:>+8.1f}', flush=True)
            random_results = evaluate(model, env, randoms) if randoms else []
            summaries.append((label, summarize(door_results),
                              summarize(random_results) if randoms else None))
    finally:
        env.close()

    def row(name, s):
        return (f'{name:44} {s["success"]:>7.1%} {s["collision"]:>9.1%} {s["timeout"]:>7.1%} '
                f'{s["time_s"]:>7.1f} {s["path_ratio"]:>6.2f} {s["reverse"]:>7.1%} '
                f'{s["return"]:>+8.1f}' + (f' {s["override"]:>8.1%}' if args.safety else ''))
    header = (f'{"":44} {"success":>7} {"collision":>9} {"timeout":>7} '
              f'{"time_s":>7} {"path":>6} {"reverse":>7} {"return":>8}'
              + (f' {"override":>8}' if args.safety else ''))
    print(f'\nDoors from spawn ({len(doors)} episodes per model, {mode})')
    print(header)
    for label, door_summary, _ in summaries:
        print(row(label, door_summary))
    if randoms:
        print(f'\nHeld-out random pairs ({len(randoms)} episodes per model, {mode})')
        print(header)
        for label, _, random_summary in summaries:
            print(row(label, random_summary))
    print('\ntime_s and path (driven / geodesic start distance) average successful '
          'episodes only; reverse = share of steps commanding reverse'
          + ('; override = share of steps the safety layer changed the command.'
             if args.safety else '.'))


if __name__ == '__main__':
    main()
