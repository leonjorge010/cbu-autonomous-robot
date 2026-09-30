"""Manual smoke test: drives HallwayEnv with random actions for a few episodes.

Run against a live sim (see launch/train_env.launch.py) to verify reset(),
step(), and the /world/hallway/control service bridge all work before
wiring up an actual training loop.
"""
from cash_rl.hallway_env import HallwayEnv


def main():
    env = HallwayEnv()
    try:
        for episode in range(3):
            obs, _ = env.reset()
            episode_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
            print(f'episode {episode}: steps={steps} reward={episode_reward:.2f} '
                  f'ended_by={"collision" if terminated else "truncated"}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
