"""Interactive demo: place the robot, pick a destination, watch the trained policy drive.

    ros2 run cash_rl demo                                   # newest runs/*/best_model.zip
    ros2 run cash_rl demo --policy ~/ros2_ws/runs/<run>/best_model.zip

In the window:
  left-click            place the robot, then left-click again to aim it
  1-7                   drive to that door (numbers are shown on the map)
  right-click           drive to the clicked point instead
  Esc                   clear and place the robot again
After an episode ends, pick another destination to continue from where the
robot stopped (an escort with several legs), or left-click to start over.
Runs HallwayRaycastEnv (the 2D simulator); no ROS or Gazebo needed.
"""
import argparse
import glob
import os

from cash_rl import spec
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.train import runs_dir
from cash_rl.view_env import load_trained_policy, Scene
import numpy as np

DOOR_KEYS = {str(i + 1): name for i, name in enumerate(sorted(spec.DOOR_GOALS))}
HELP = ('left-click: place robot, click again to aim  ·  1-7: door  ·  '
        'right-click: drive to point  ·  Esc: start over')


def newest_best_model():
    models = glob.glob(os.path.join(runs_dir(), '*', 'best_model.zip'))
    if not models:
        raise SystemExit(f'No best_model.zip under {runs_dir()}; train one or pass --policy')
    return max(models, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--policy', help='train.py model .zip (default: newest best_model.zip)')
    parser.add_argument('--speed', type=float, default=1.5, help='playback speed multiplier')
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    model_path = os.path.expanduser(args.policy or newest_best_model())
    policy = load_trained_policy(model_path)
    env = HallwayRaycastEnv()
    model_name = os.path.relpath(model_path, os.path.dirname(runs_dir()))

    fig, ax = plt.subplots(figsize=(14, 6.6))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.1)
    fig.canvas.manager.set_window_title(f'C.A.$.H demo — {model_name}')
    scene = Scene(ax, env, door_labels={name: f'{key}  {name}'
                                        for key, name in DOOR_KEYS.items()})
    fig.text(0.04, 0.015, HELP, fontsize=9, color='0.35')

    # mode: 'place' -> 'aim' -> 'ready' -> 'driving' -> 'done' (-> 'driving' again)
    st = {'mode': 'place', 'pose': None, 'ranges': None, 'goal': None, 'trail': [],
          'goal_name': '', 'step': 0, 'return': 0.0, 'episode': 0,
          'text': 'Left-click to place the robot'}

    def redraw():
        scene.draw(st['pose'], st['ranges'], st['goal'], st['trail'], st['text'])
        fig.canvas.draw_idle()

    def preview(pose):
        st['pose'] = np.asarray(pose, dtype=float)
        st['ranges'] = spec.sanitize_ranges(env.raycast(st['pose']))

    def start_over(text='Left-click to place the robot'):
        st.update(mode='place', pose=None, ranges=None, goal=None, trail=[], text=text)

    def drive_to(goal, name):
        if st['mode'] not in ('ready', 'done'):
            return
        if env.collides(st['pose']):
            start_over('The robot is touching a wall — left-click to place it again')
            return
        st['episode'] += 1
        try:
            env.reset(seed=st['episode'],
                      options={'spawn': tuple(st['pose']), 'goal': tuple(goal)})
        except RuntimeError:
            st['text'] = f"Can't drive to {name}: not reachable from here. Pick another."
            return
        st.update(mode='driving', goal=env.goal.copy(), goal_name=name, step=0,
                  trail=[env.pose[:2].copy()], ranges=env._ranges.copy())
        st['return'] = 0.0

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if fig.canvas.toolbar is not None and str(fig.canvas.toolbar.mode):
            return  # zoom/pan tool active
        xy = np.array([event.xdata, event.ydata])
        if event.button == 3:
            if not env.map.is_free(*xy):
                st['text'] = 'That spot is inside a wall — right-click open floor'
            else:
                drive_to(xy, f'point ({xy[0]:.1f}, {xy[1]:.1f})')
            redraw()
            return
        if event.button != 1:
            return
        if st['mode'] in ('place', 'ready', 'done'):
            if not env.map.is_free(*xy):
                st['text'] = 'That spot is inside a wall — click open floor'
            else:
                start_over()
                preview((*xy, 0.0))
                st.update(mode='aim', text='Left-click again to set which way it faces')
        elif st['mode'] == 'aim':
            dx, dy = xy - st['pose'][:2]
            if np.hypot(dx, dy) > 1e-3:
                preview((*st['pose'][:2], np.arctan2(dy, dx)))
            if env.collides(st['pose']):
                start_over('Too close to a wall at that angle — left-click to place again')
            else:
                st.update(mode='ready', trail=[],
                          text='Press 1-7 for a door, or right-click a destination')
        redraw()

    def on_move(event):
        if st['mode'] != 'aim' or event.inaxes is not ax or event.xdata is None:
            return
        dx, dy = np.array([event.xdata, event.ydata]) - st['pose'][:2]
        if np.hypot(dx, dy) > 1e-3:
            preview((*st['pose'][:2], np.arctan2(dy, dx)))
            redraw()

    def on_key(event):
        if event.key == 'escape':
            start_over()
        elif event.key in DOOR_KEYS and st['mode'] in ('ready', 'done'):
            name = DOOR_KEYS[event.key]
            drive_to(spec.DOOR_GOALS[name], name)
        redraw()

    def tick():
        if st['mode'] != 'driving':
            return
        _, reward, terminated, truncated, info = env.step(policy(env))
        st['step'] += 1
        st['return'] += reward
        st['pose'], st['ranges'] = env.pose.copy(), env._ranges.copy()
        st['trail'].append(env.pose[:2].copy())
        outcome = ''
        if terminated or truncated:
            outcome = ('REACHED GOAL' if info['is_success']
                       else 'COLLISION' if info['collided'] else 'TIMED OUT')
            st['mode'] = 'done'
        st['text'] = (f'goal: {st["goal_name"]}   step {st["step"]}/{env._max_steps}   '
                      f'{st["step"] * spec.CONTROL_PERIOD:.1f} s   '
                      f'{info["goal_distance"]:.1f} m to go   {outcome}')
        if outcome:
            st['text'] += '  — pick another destination, or left-click to start over'
        redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    timer = fig.canvas.new_timer(interval=int(1000 * spec.CONTROL_PERIOD / args.speed))
    timer.add_callback(tick)
    timer.start()
    redraw()
    print(f'Demo using {model_path}')
    plt.show()


if __name__ == '__main__':
    main()
