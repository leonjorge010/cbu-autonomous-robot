"""Evaluate a trained policy on the door goals, in the 2D env or in Gazebo.

    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip              # 2D
    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip --env gazebo

--env gazebo needs `ros2 launch cash_rl train_env.launch.py` running, and gives
the most faithful result when nothing else is loading the CPU: HallwayEnv steps
on wall-clock time, so a slow simulation changes the sim time per step.
Every episode starts at spec.SPAWN_POSE; prints one row per door plus totals.
"""
import argparse

from cash_rl import spec
from cash_rl.train import evaluate, load_model
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model', help='train.py model .zip')
    parser.add_argument('--env', choices=['2d', 'gazebo'], default='2d')
    parser.add_argument('--repeats', type=int, default=1, help='episodes per door')
    args = parser.parse_args()

    model = load_model(args.model)
    if args.env == 'gazebo':
        from cash_rl.hallway_env import HallwayEnv
        env = HallwayEnv()
    else:
        from cash_rl.raycast_env import HallwayRaycastEnv
        env = HallwayRaycastEnv()
    episodes = [(name, repeat, {'spawn': spec.SPAWN_POSE, 'goal': goal})
                for name, goal in sorted(spec.DOOR_GOALS.items())
                for repeat in range(args.repeats)]
    try:
        print(f'{"door":10} {"outcome":10} {"steps":>6} {"return":>8}', flush=True)
        results = []
        for episode in episodes:
            result = evaluate(model, env, [episode])[0]
            results.append(result)
            outcome = ('reached' if result['success'] else
                       'collision' if result['collision'] else 'timeout')
            print(f'{result["name"]:10} {outcome:10} {result["steps"]:>6} '
                  f'{result["return"]:>+8.1f}', flush=True)
    finally:
        env.close()

    def rate(key):
        return np.mean([r[key] for r in results])
    print(f'\n{args.env}: success {rate("success"):.0%}  collision {rate("collision"):.0%}  '
          f'timeout {rate("timeout"):.0%}  mean return {rate("return"):+.1f}')


if __name__ == '__main__':
    main()
