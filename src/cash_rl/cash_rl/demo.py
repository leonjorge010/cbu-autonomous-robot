"""Interactive demo: place the robot, pick a destination, watch the trained policy drive.

    ros2 run cash_rl demo                                   # newest runs/*/best_model.zip
    ros2 run cash_rl demo --policy ~/ros2_ws/runs/<run>/best_model.zip

In the window:
  left-click            place the robot (facing +x, down the corridor)
  left-drag             place it and aim it: press at the spot, release toward
                        the direction it should face
  right-click           drive to the clicked point
  1-7                   drive to that door (numbers are shown on the map)
  Esc                   clear the robot
After an episode ends, right-click (or 1-7) again to continue from where the
robot stopped (an escort with several legs), or left-click to place it anew.
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
HELP = ('left-click: place robot (drag to aim)  ·  right-click: destination  ·  '
        '1-7: door  ·  Esc: clear')
# A left-drag shorter than this (m) counts as a plain click: keep the default heading.
MIN_AIM_DRAG = 0.2


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
    fig.canvas.manager.set_window_title(f'CBU Autonomous Robot demo — {model_name}')
    scene = Scene(ax, env, door_labels={name: f'{key}  {name}'
                                        for key, name in DOOR_KEYS.items()})
    fig.text(0.04, 0.015, HELP, fontsize=9, color='0.35')

    # mode: 'place' -> 'aim' (left button held) -> 'ready' -> 'driving' -> 'done'
    st = {'mode': 'place', 'pose': None, 'ranges': None, 'goal': None, 'trail': [],
          'goal_name': '', 'step': 0, 'return': 0.0, 'episode': 0,
          'text': 'Left-click to place the robot (drag to aim it)'}

    def redraw():
        scene.draw(st['pose'], st['ranges'], st['goal'], st['trail'], st['text'])
        fig.canvas.draw_idle()

    def preview(pose):
        st['pose'] = np.asarray(pose, dtype=float)
        st['ranges'] = spec.sanitize_ranges(env.raycast(st['pose']))

    def start_over(text='Left-click to place the robot (drag to aim it)'):
        st.update(mode='place', pose=None, ranges=None, goal=None, trail=[], text=text)

    def drive_to(goal, name):
        if st['mode'] not in ('ready', 'done'):
            return
        if env.collides(st['pose']):
            start_over('The robot is touching a wall — left-click to place it again')
            return
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

    def axes_point(event):
        if event.inaxes is not ax or event.xdata is None:
            return None
        if fig.canvas.toolbar is not None and str(fig.canvas.toolbar.mode):
            return None  # zoom/pan tool active
        return np.array([event.xdata, event.ydata])

    def aim_at(xy):
        dx, dy = xy - st['pose'][:2]
        if np.hypot(dx, dy) >= MIN_AIM_DRAG:
            preview((*st['pose'][:2], np.arctan2(dy, dx)))

    def on_press(event):
        xy = axes_point(event)
        if xy is None:
            return
        if event.button == 3:
            if st['mode'] not in ('ready', 'done'):
                st['text'] = 'Left-click to place the robot first'
            elif not env.map.is_free(*xy):
                st['text'] = 'That spot is inside a wall — right-click open floor'
            else:
                drive_to(xy, f'point ({xy[0]:.1f}, {xy[1]:.1f})')
        elif event.button == 1 and st['mode'] != 'driving':
            if not env.map.is_free(*xy):
                st['text'] = 'That spot is inside a wall — left-click open floor'
            else:
                start_over()
                preview((*xy, 0.0))
                st.update(mode='aim', text='Release to place; drag first to aim it')
        redraw()

    def on_move(event):
        xy = axes_point(event)
        if st['mode'] == 'aim' and xy is not None:
            aim_at(xy)
            redraw()

    def on_release(event):
        if event.button != 1 or st['mode'] != 'aim':
            return
        xy = axes_point(event)
        if xy is not None:
            aim_at(xy)
        if env.collides(st['pose']):
            start_over('Too close to a wall there — left-click to place it again')
        else:
            st.update(mode='ready', trail=[],
                      text='Right-click a destination, or press 1-7 for a door')
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
            st['text'] += '  — right-click to continue, or left-click to place anew'
        redraw()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    timer = fig.canvas.new_timer(interval=int(1000 * spec.CONTROL_PERIOD / args.speed))
    timer.add_callback(tick)
    timer.start()
    redraw()
    print(f'Demo using {model_path}')
    plt.show()


if __name__ == '__main__':
    main()
