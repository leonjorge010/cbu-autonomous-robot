"""Evaluate trained policies on the door goals (+ held-out random goals), 2D or Gazebo.

    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip              # 2D, doors
    ros2 run cash_rl evaluate ~/ros2_ws/runs/<run>/best_model.zip --env gazebo
    ros2 run cash_rl evaluate runs/a/best_model.zip runs/b/best_model.zip --random 200
    ros2 run cash_rl evaluate runs/a/best_model.zip --random 200 --pedestrians 4
    ros2 run cash_rl evaluate downhill avoider --random 200 --pedestrians 4  # 2D only
    ros2 run cash_rl evaluate runs/a/final_model.zip --map cbu_floor2 --random 200 \
        --subgoal-lookahead 3                     # policy sees a point 3 m ahead on the route
    ros2 run cash_rl evaluate runs/a/final_model.zip avoider --random 200 --lane-offset 0.5
                                                  # subgoal pinned 0.5 m left of center

--env gazebo needs `ros2 launch cash_rl train_env.launch.py` running, and gives
the most faithful result when nothing else is loading the CPU: HallwayEnv steps
on wall-clock time, so a slow simulation changes the sim time per step.
Door episodes start at spec.SPAWN_POSE. --random N adds N random spawn/goal
pairs (2D only) on seeds held out from both training and train.py's periodic
eval, which picks best_model.zip and so would flatter it. Several models are
scored on identical episodes and summarized side by side. The scripted
controllers `downhill` and `avoider` (cash_rl.scripted) can be named in place of
a model zip as references; they read the 2D env's state, so 2D only.
"""
import argparse
import os

from cash_rl import scripted, spec
from cash_rl.occupancy_map import default_map_yaml, default_map_yaml_in_workspace, use_map
from cash_rl.train import EVAL_SEED_BASE, evaluate, load_model, model_map_yaml
from cash_rl.train import model_subgoal_lookahead, model_subgoal_offset
import numpy as np

# Held-out random pairs: past train.py's periodic-eval seeds (EVAL_SEED_BASE + i).
HELD_OUT_SEED_BASE = EVAL_SEED_BASE + 100_000
SCRIPTED = {'downhill': scripted.downhill, 'avoider': scripted.avoider}


class ScriptedModel:
    """predict() adapter for a cash_rl.scripted controller reading the env's state."""

    def __init__(self, controller, env):
        self.controller = controller
        self.env = env.unwrapped

    def predict(self, observation, **kwargs):
        return self.controller(self.env), None


def summarize(results):
    def mean(key, subset):
        return float(np.mean([r[key] for r in subset])) if subset else float('nan')
    ok = [r for r in results if r['success']]
    lane_errors = [e for r in results for e in r['lane_errors']]
    return {
        'success': mean('success', results),
        'collision': mean('collision', results),
        'timeout': mean('timeout', results),
        'time_s': mean('steps', ok) * spec.CONTROL_PERIOD,
        'path_ratio': float(np.mean([r['path'] / r['start_distance'] for r in ok]))
        if ok else float('nan'),
        'reverse': mean('reverse', results),
        'override': mean('override', results),
        'lane_error': float(np.mean(lane_errors)) if lane_errors else float('nan'),
        'return': mean('return', results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('models', nargs='+',
                        help=f"train.py model .zip files, or {' / '.join(SCRIPTED)}")
    parser.add_argument('--env', choices=['2d', 'gazebo'], default='2d')
    parser.add_argument('--repeats', type=int, default=1, help='episodes per door')
    parser.add_argument('--random', type=int, default=0,
                        help='held-out random spawn/goal pairs to add (2D only)')
    parser.add_argument('--pedestrians', type=int, default=0,
                        help='moving pedestrians per episode (2D only)')
    parser.add_argument('--yielding-pedestrians', action='store_true',
                        help='pedestrians step around the robot instead of walking into it')
    parser.add_argument('--encounter-prob', type=float, default=0.0,
                        help="share of episodes whose first pedestrian starts on the robot's "
                             'route walking toward it; 1 = an avoidance test (2D only)')
    parser.add_argument('--map', help='map name in maps/ or .yaml path (default: the map the '
                        'first model zip was trained on, else $CASH_MAP_YAML / hallway); '
                        'door episodes are hallway only')
    parser.add_argument('--subgoal-lookahead', type=float, default=None, metavar='M',
                        help='observe a subgoal M m ahead on the geodesic route instead '
                             'of the goal (2D only; reward and success still use the goal; '
                             "default: the first model zip's training setting; 0 = off)")
    parser.add_argument('--subgoal-offset', type=float, default=None, metavar='W',
                        help='subgoal in a random lane up to W m left/right of the corridor '
                             "center (2D only; default: the first model zip's training "
                             'setting; 0 = off)')
    parser.add_argument('--lane-offset', type=float, default=None, metavar='X',
                        help='pin the subgoal lane X m left (+) / right (-) of center for '
                             'every episode (2D only; needs a subgoal lookahead)')
    parser.add_argument('--safety', action='store_true',
                        help='run the policy behind the LiDAR safety layer (cash_rl.safety)')
    args = parser.parse_args()
    if args.random and args.env == 'gazebo':
        parser.error('--random is 2D only (HallwayEnv spawns from fixed poses)')
    trained_lookahead = [model_subgoal_lookahead(p) for p in args.models if p not in SCRIPTED]
    if args.subgoal_lookahead is None and trained_lookahead and args.env == '2d':
        args.subgoal_lookahead = trained_lookahead[0]
    for path, lookahead in zip([p for p in args.models if p not in SCRIPTED], trained_lookahead):
        if (lookahead or None) != (args.subgoal_lookahead or None):
            print(f'WARNING: {path} was trained with subgoal lookahead {lookahead}, '
                  f'evaluating with {args.subgoal_lookahead or None}', flush=True)
    if args.subgoal_lookahead and args.env == 'gazebo':
        parser.error("--subgoal-lookahead is 2D only (it reads the 2D map's route)")
    trained_offset = [model_subgoal_offset(p) for p in args.models if p not in SCRIPTED]
    if args.subgoal_offset is None:
        args.subgoal_offset = trained_offset[0] if trained_offset and args.env == '2d' else 0.0
    if args.lane_offset is not None:
        args.subgoal_offset = 0.0  # pinned instead of drawn
    if (args.subgoal_offset or args.lane_offset is not None) and not args.subgoal_lookahead:
        parser.error('--subgoal-offset / --lane-offset need a subgoal lookahead')
    if args.pedestrians and args.env == 'gazebo':
        parser.error('--pedestrians is 2D only (no pedestrians in Gazebo yet)')
    if args.env == 'gazebo' and any(path in SCRIPTED for path in args.models):
        parser.error(f"{' / '.join(SCRIPTED)} are 2D only (they read the 2D env's state)")

    trained_on = [model_map_yaml(p) for p in args.models if p not in SCRIPTED]
    if args.map:
        use_map(args.map)
    elif trained_on and trained_on[0]:
        use_map(trained_on[0])
    map_yaml = os.path.abspath(default_map_yaml())
    for path, on in zip([p for p in args.models if p not in SCRIPTED], trained_on):
        if os.path.realpath(on or default_map_yaml_in_workspace()) != os.path.realpath(map_yaml):
            print(f'WARNING: {path} was trained on {on or "the hallway"}, '
                  f'evaluating on {map_yaml}', flush=True)

    if args.env == 'gazebo':
        from cash_rl.hallway_env import HallwayEnv
        env = HallwayEnv()
    else:
        from cash_rl.raycast_env import HallwayRaycastEnv
        env = HallwayRaycastEnv(n_pedestrians=args.pedestrians,
                                pedestrians_yield=args.yielding_pedestrians,
                                encounter_prob=args.encounter_prob,
                                subgoal_lookahead=args.subgoal_lookahead,
                                subgoal_offset=args.subgoal_offset)
    if args.safety:
        from cash_rl.safety import SafetyWrapper
        env = SafetyWrapper(env)
    mode = args.env + ('' if args.env == 'gazebo'
                       else f', {os.path.splitext(os.path.basename(map_yaml))[0]}') \
        + (f', {args.pedestrians} pedestrians' if args.pedestrians else '') \
        + (' (yielding)' if args.pedestrians and args.yielding_pedestrians else '') \
        + (f', encounters {args.encounter_prob:g}' if args.pedestrians and args.encounter_prob
           else '') \
        + (f', subgoals {args.subgoal_lookahead:g} m' if args.subgoal_lookahead else '') \
        + (f', lanes +-{args.subgoal_offset:g} m' if args.subgoal_offset else '') \
        + (f', lane {args.lane_offset:+g} m' if args.lane_offset is not None else '') \
        + (', safety layer' if args.safety else '')
    door_goals = spec.DOOR_GOALS if args.env == 'gazebo' else env.unwrapped.door_goals
    doors = [(name, repeat, {'spawn': spec.SPAWN_POSE, 'goal': goal})
             for name, goal in sorted(door_goals.items())
             for repeat in range(args.repeats)]
    if not doors and not args.random:
        parser.error('this map has no door episodes; pass --random N')
    randoms = [(f'random_{i}', HELD_OUT_SEED_BASE + i, None) for i in range(args.random)]
    if args.lane_offset is not None:
        doors = [(n, r, dict(o, lane_offset=args.lane_offset)) for n, r, o in doors]
        randoms = [(n, s, {'lane_offset': args.lane_offset}) for n, s, _ in randoms]

    summaries = []
    try:
        for path in args.models:
            if path in SCRIPTED:
                model, label = ScriptedModel(SCRIPTED[path], env), path
            else:
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

    lanes = bool(args.subgoal_offset) or args.lane_offset is not None

    def row(name, s):
        return (f'{name:44} {s["success"]:>7.1%} {s["collision"]:>9.1%} {s["timeout"]:>7.1%} '
                f'{s["time_s"]:>7.1f} {s["path_ratio"]:>6.2f} {s["reverse"]:>7.1%} '
                f'{s["return"]:>+8.1f}' + (f' {s["override"]:>8.1%}' if args.safety else '')
                + (f' {s["lane_error"]:>6.2f}' if lanes else ''))
    header = (f'{"":44} {"success":>7} {"collision":>9} {"timeout":>7} '
              f'{"time_s":>7} {"path":>6} {"reverse":>7} {"return":>8}'
              + (f' {"override":>8}' if args.safety else '')
              + (f' {"lane":>6}' if lanes else ''))
    if doors:
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
          + ('; override = share of steps the safety layer changed the command'
             if args.safety else '')
          + ('; lane = mean |sideways distance| (m) from the subgoal lane' if lanes else '')
          + '.')


if __name__ == '__main__':
    main()
