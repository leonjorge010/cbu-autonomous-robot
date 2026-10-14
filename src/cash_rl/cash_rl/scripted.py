"""Hand-written reference controller for HallwayRaycastEnv.

Not a learned policy: it cheats by reading the env's map and geodesic distance
field (which the observation doesn't include). Used by tests to check that every
goal is reachable, and by view_env as a baseline to compare trained policies
against.
"""
import numpy as np


def downhill(env, lookahead=0.6, wall_clearance=0.7, wall_weight=3.0):
    """Head down the geodesic goal distance, keeping clear of walls; return an action."""
    x, y, yaw = env.pose
    to_goal = env.goal - env.pose[:2]
    if np.hypot(*to_goal) < lookahead:
        heading = np.arctan2(to_goal[1], to_goal[0])
    else:
        dirs = np.linspace(-np.pi, np.pi, 48, endpoint=False)
        px, py = x + lookahead * np.cos(dirs), y + lookahead * np.sin(dirs)
        cost = np.array([env._goal_field(a, b) for a, b in zip(px, py)])
        cost = np.where(np.isfinite(cost), cost, np.inf)
        row, col = env.map.to_cell(px, py)
        clearance = env.map.clearance[np.clip(row, 0, env.map.height - 1),
                                      np.clip(col, 0, env.map.width - 1)]
        cost = cost + wall_weight * np.maximum(0.0, wall_clearance - clearance)
        heading = dirs[np.argmin(cost)]
    error = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([1.0 if abs(error) < 0.4 else 0.1, np.clip(2 * error, -1, 1)])
