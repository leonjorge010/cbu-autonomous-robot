"""Watch HallwayRaycastEnv episodes: live in a window, or saved as a GIF.

    ros2 run cash_rl view_env                        # downhill controller, random goals
    ros2 run cash_rl view_env --policy random --goal door_201
    ros2 run cash_rl view_env --goal all --save doors.gif
    ros2 run cash_rl view_env --policy ~/ros2_ws/runs/<run>/best_model.zip --goal all

Shows the SLAM map to scale with the robot footprint (collision margin
included), the 36 observed lidar beams, the goal tolerance circle, the path so
far, and episode step/return/outcome. Episodes are simulated up front (the 2D
env is fast), then played back at real time × --speed.
"""
import argparse

from cash_rl import spec
from cash_rl.raycast_env import HallwayRaycastEnv
from cash_rl.scripted import downhill
import numpy as np

HOLD_FRAMES = 15  # Pause on each episode's last frame.
POLICIES = {
    'downhill': downhill,
    'random': lambda env: env.action_space.sample(),
}


def load_trained_policy(path):
    """Policy from a train.py model zip."""
    from cash_rl.train import load_model
    model = load_model(path)

    def policy(env):
        obs = spec.build_observation(env._ranges, env.pose, env.goal, env._prev_action)
        return model.predict(obs, deterministic=True)[0]
    return policy


def run_episodes(env, policy, goals, seed):
    """Simulate episodes; return a list of per-step snapshots for playback."""
    frames = []
    for i, (goal_name, goal) in enumerate(goals):
        options = {'spawn': spec.SPAWN_POSE, 'goal': goal} if goal is not None else None
        _, info = env.reset(seed=seed + i, options=options)
        goal_name = goal_name or f'random ({info["goal"][0]:.1f}, {info["goal"][1]:.1f})'
        trail, total, step, outcome = [env.pose[:2].copy()], 0.0, 0, ''
        while True:
            frames.append({
                'pose': env.pose.copy(), 'ranges': env._ranges.copy(),
                'goal': env.goal.copy(), 'trail': np.array(trail), 'text': (
                    f'episode {i + 1}/{len(goals)}   goal: {goal_name}   '
                    f'step {step}/{env._max_steps}   return {total:+.1f}   {outcome}')})
            if outcome:
                frames.extend([frames[-1]] * HOLD_FRAMES)
                break
            _, reward, terminated, truncated, info = env.step(policy(env))
            total += reward
            step += 1
            trail.append(env.pose[:2].copy())
            if terminated or truncated:
                outcome = ('REACHED GOAL' if info['is_success']
                           else 'COLLISION' if info['collided'] else 'TIMED OUT')
    return frames


def footprint_corners(pose):
    hl = spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    hw = spec.FOOTPRINT_HALF_WIDTH + spec.COLLISION_MARGIN
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pose[:2] + corners @ np.array([[c, s], [-s, c]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--policy', default='downhill',
                        help=f"{' or '.join(sorted(POLICIES))}, or a trained model .zip")
    parser.add_argument('--goal', default='random',
                        help="'random', 'all' (every door from spawn), or a door name: "
                             + ', '.join(sorted(spec.DOOR_GOALS)))
    parser.add_argument('--episodes', type=int, default=3,
                        help='number of episodes with --goal random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed', type=float, default=2.0, help='playback speed multiplier')
    parser.add_argument('--save', metavar='PATH.gif', help='write a GIF instead of a window')
    args = parser.parse_args()

    if args.goal == 'random':
        goals = [(None, None)] * args.episodes
    elif args.goal == 'all':
        goals = sorted(spec.DOOR_GOALS.items())
    elif args.goal in spec.DOOR_GOALS:
        goals = [(args.goal, spec.DOOR_GOALS[args.goal])]
    else:
        parser.error(f'unknown goal {args.goal!r}')

    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon
    import matplotlib.pyplot as plt

    env = HallwayRaycastEnv()
    policy = POLICIES.get(args.policy) or load_trained_policy(args.policy)
    frames = run_episodes(env, policy, goals, args.seed)

    m = env.map
    fig, ax = plt.subplots(figsize=(14, 6.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.07)
    ax.imshow(m.pixels, cmap='gray', origin='lower', vmin=0, vmax=255, extent=[
        m.origin[0], m.origin[0] + m.width * m.resolution,
        m.origin[1], m.origin[1] + m.height * m.resolution])
    for name, (gx, gy) in spec.DOOR_GOALS.items():
        ax.plot(gx, gy, '+', color='0.6', ms=6)
        ax.annotate(name, (gx, gy), xytext=(3, 4), textcoords='offset points',
                    fontsize=7, color='0.45')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    beams = ax.add_collection(LineCollection([], colors='tab:red', linewidths=0.6, alpha=0.6))
    trail, = ax.plot([], [], color='tab:blue', lw=1.2, alpha=0.7)
    robot = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc='tab:blue', ec='navy',
                                 alpha=0.8))
    heading, = ax.plot([], [], color='white', lw=2)
    goal = ax.add_patch(Circle((0, 0), spec.GOAL_TOLERANCE, fc='orange', ec='darkorange',
                               alpha=0.7))
    title = ax.set_title('', fontsize=11, family='monospace')

    def draw(frame):
        pose, ranges = frame['pose'], frame['ranges']
        c, s = np.cos(pose[2]), np.sin(pose[2])
        lidar = pose[:2] + spec.LIDAR_X * np.array([c, s])
        angles = pose[2] + spec.OBS_BEAM_ANGLES
        ends = lidar + ranges[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        beams.set_segments([[lidar, end] for end in ends])
        trail.set_data(frame['trail'][:, 0], frame['trail'][:, 1])
        robot.set_xy(footprint_corners(pose))
        tip = pose[:2] + spec.FOOTPRINT_HALF_LENGTH * np.array([c, s])
        heading.set_data([pose[0], tip[0]], [pose[1], tip[1]])
        goal.center = frame['goal']
        title.set_text(frame['text'])
        return beams, trail, robot, heading, goal, title

    interval_ms = 1000 * spec.CONTROL_PERIOD / args.speed
    anim = FuncAnimation(fig, draw, frames=frames, interval=interval_ms, blit=False,
                         repeat=not args.save, cache_frame_data=False)
    if args.save:
        anim.save(args.save, writer=PillowWriter(fps=round(1000 / interval_ms)), dpi=70)
        print(f'wrote {args.save} ({len(frames)} frames)')
    else:
        plt.show()


if __name__ == '__main__':
    main()
