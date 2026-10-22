"""Hand-written reference controllers for HallwayRaycastEnv.

Not learned policies: they cheat by reading the env's map and geodesic distance
field (which the observation doesn't include); avoider also reads the true
pedestrian positions and velocities. Used by tests to check that every
goal is reachable, and by view_env as a baseline to compare trained policies
against.
"""
from cash_rl import spec
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


def avoider(env, horizon=4.0, wall_horizon=1.0, hard_horizon=1.5, comfort=0.4,
            pedestrian_weight=4.0, contact_cost=20.0, wall_clearance=0.7, wall_weight=1.0):
    """Downhill plus pedestrian avoidance from their true future positions; return an action.

    Cheats more than downhill: besides the geodesic field it reads the pedestrians'
    positions and velocities and predicts their exact paths (non-yielding
    pedestrians ignore the robot). Each step it rolls out a grid of constant
    actions over `horizon` seconds and scores each by its best estimated arrival
    time if it switched back to heading downhill at some step (before any
    contact). It skips actions that hit a wall within wall_horizon or a pedestrian
    within hard_horizon, and charges contact_cost for a pedestrian hit later in the
    horizon plus pedestrian_weight per second within `comfort` of one (both in
    metres of geodesic, converted to time at full speed). An upper-bound reference
    for what avoidance is achievable with this robot, not something a policy could
    copy from its observation.
    """
    base = env.unwrapped
    if len(base.pedestrians) == 0 or base.pedestrians_yield:
        return downhill(base)
    dt = spec.CONTROL_PERIOD
    n_steps = int(round(horizon / dt))
    future = _future_pedestrians(base, n_steps)  # (T, N, 2)

    linear, angular = np.meshgrid([-1.0, -0.5, 0.0, 0.3, 0.6, 1.0], np.linspace(-1, 1, 9))
    actions = np.stack([linear.ravel(), angular.ravel()], axis=1)  # (K, 2)
    targets = np.array([spec.action_to_cmd(a) for a in actions])
    max_delta = np.array([spec.MAX_LINEAR_ACCEL, spec.MAX_ANGULAR_ACCEL]) * dt
    velocity = np.tile(base._velocity, (len(actions), 1))
    pose = np.tile(base.pose, (len(actions), 1))
    poses = np.empty((n_steps, len(actions), 3))
    for t in range(n_steps):
        velocity += np.clip(targets - velocity, -max_delta, max_delta)
        pose[:, 0] += velocity[:, 0] * np.cos(pose[:, 2]) * dt
        pose[:, 1] += velocity[:, 0] * np.sin(pose[:, 2]) * dt
        pose[:, 2] += velocity[:, 1] * dt
        poses[t] = pose

    reached = np.hypot(poses[..., 0] - base.goal[0],
                       poses[..., 1] - base.goal[1]) < spec.GOAL_TOLERANCE
    active = np.cumsum(reached, axis=0) == 0  # (T, K): before reaching the goal
    gaps = _pedestrian_gaps(poses, future).min(axis=2)  # (T, K)
    wall_steps = int(round(wall_horizon / dt))
    hits_wall = np.zeros((n_steps, len(actions)), dtype=bool)
    for t in range(wall_steps):
        hits_wall[t] = _walls_hit(base, base._collision_outline, poses[t])
    contact = active & ((gaps < 0.0) | hits_wall)
    first = np.where(np.any(contact, axis=0), np.argmax(contact, axis=0), n_steps)  # (K,)
    blocked = (first + 1) * dt <= hard_horizon
    discomfort = np.sum(active * np.maximum(0.0, comfort - gaps), axis=0) * dt / comfort
    later_contact = np.any(active & (gaps < 0.0), axis=0)

    # Estimated arrival time if the robot switched back to heading downhill after
    # each rollout step: time so far + geodesic distance at full speed + the turn
    # still needed toward the descent direction. Each candidate scores its best.
    field = base._goal_field.grid
    row, col = base.map.to_cell(poses[..., 0], poses[..., 1])
    inside = (row >= 0) & (row < base.map.height) & (col >= 0) & (col < base.map.width)
    to_go = np.full(row.shape, np.inf)
    to_go[inside] = field[row[inside], col[inside]]
    to_go = np.where(np.isfinite(to_go), to_go, 1e3)
    heading = _descent_heading(base)
    turn = np.abs((heading - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi)
    clearance = base._clearance_at(poses[..., 0], poses[..., 1])
    times = (np.arange(n_steps) + 1)[:, None] * dt
    eta = times + to_go / spec.MAX_LINEAR_VEL + turn / spec.MAX_ANGULAR_VEL \
        + wall_weight * np.maximum(0.0, wall_clearance - clearance) / spec.MAX_LINEAR_VEL
    eta = np.where(reached, times, eta)
    before_contact = np.arange(n_steps)[:, None] < first
    eta = np.where((active | reached) & before_contact, eta, np.inf)
    cost = eta.min(axis=0) + (pedestrian_weight * discomfort
                              + contact_cost * later_contact) / spec.MAX_LINEAR_VEL
    if np.all(blocked):
        # Nothing is safe: take the action whose first contact comes latest.
        return actions[np.argmax(first - 0.01 * cost)]
    return actions[np.argmin(np.where(blocked, np.inf, cost))]


def _descent_heading(env, lookahead=0.6):
    """Heading (rad) of steepest geodesic descent from the robot, or toward a close goal."""
    to_goal = env.goal - env.pose[:2]
    if np.hypot(*to_goal) < lookahead:
        return np.arctan2(to_goal[1], to_goal[0])
    dirs = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    px, py = env.pose[0] + lookahead * np.cos(dirs), env.pose[1] + lookahead * np.sin(dirs)
    cost = np.array([env._goal_field(a, b) for a, b in zip(px, py)])
    return dirs[np.argmin(np.where(np.isfinite(cost), cost, np.inf))]


def _future_pedestrians(env, n_steps):
    """(n_steps, N, 2) pedestrian positions after each of the next steps; env unchanged."""
    saved = env.pedestrians.copy(), env._pedestrian_velocities.copy()
    future = np.empty((n_steps,) + env.pedestrians.shape)
    try:
        for t in range(n_steps):
            env._move_pedestrians(spec.CONTROL_PERIOD)
            future[t] = env.pedestrians
    finally:
        env.pedestrians, env._pedestrian_velocities = saved
    return future


def _pedestrian_gaps(poses, pedestrians):
    """(T, K, N) edge gaps between the inflated footprint at poses (T, K, 3) and discs."""
    from cash_rl.raycast_env import PEDESTRIAN_RADIUS
    dx = pedestrians[:, None, :, 0] - poses[..., 0, None]
    dy = pedestrians[:, None, :, 1] - poses[..., 1, None]
    c, s = np.cos(poses[..., 2, None]), np.sin(poses[..., 2, None])
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    half_length = spec.FOOTPRINT_HALF_LENGTH + spec.COLLISION_MARGIN
    half_width = spec.FOOTPRINT_HALF_WIDTH + spec.COLLISION_MARGIN
    return np.hypot(np.maximum(np.abs(local_x) - half_length, 0.0),
                    np.maximum(np.abs(local_y) - half_width, 0.0)) - PEDESTRIAN_RADIUS


def _walls_hit(env, outline, poses):
    """(K,) True where the footprint outline at poses (K, 3) leaves free space."""
    c, s = np.cos(poses[:, 2, None]), np.sin(poses[:, 2, None])
    wx = poses[:, 0, None] + c * outline[:, 0] - s * outline[:, 1]
    wy = poses[:, 1, None] + s * outline[:, 0] + c * outline[:, 1]
    return ~np.all(env.map.is_free(wx, wy), axis=1)
