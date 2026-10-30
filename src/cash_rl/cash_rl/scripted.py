"""Hand-written reference controllers for HallwayRaycastEnv.

Not learned policies: they cheat by reading the env's map and geodesic distance
field (which the observation doesn't include); avoider also reads the true
pedestrian positions and velocities. Used by tests to check that every
goal is reachable, and by view_env as a baseline to compare trained policies
against.
"""
from cash_rl import spec
import numpy as np


def downhill(env, lookahead=0.6, wall_clearance=0.7, wall_weight=3.0, lane_weight=2.0,
             lane_deadband=0.05):
    """Head down the geodesic goal distance, keeping clear of walls; return an action.

    With a subgoal lane active (HallwayRaycastEnv subgoal_offset / lane_offset),
    each metre the lookahead point sits outside the lane (beyond lane_deadband)
    costs lane_weight metres of geodesic, so it steers into the lane and follows it.
    """
    env = env.unwrapped
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
        if env.near_lane is not None:
            error = np.abs(env.lane_error(np.stack([px, py], axis=1)))
            cost = cost + lane_weight * np.maximum(0.0, error - lane_deadband)
        heading = dirs[np.argmin(cost)]
    error = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([1.0 if abs(error) < 0.4 else 0.1, np.clip(2 * error, -1, 1)])


def avoider(env, horizon=4.0, wall_horizon=1.0, hard_horizon=1.5, comfort=0.4,
            pedestrian_weight=4.0, contact_cost=20.0, wall_clearance=0.7, wall_weight=1.0,
            side_weight=1.0, side_distance=1.0, lane_weight=0.5, lane_deadband=0.05,
            lane_release=2.0):
    """Downhill plus pedestrian avoidance from their true future positions; return an action.

    Cheats more than downhill: besides the geodesic field it reads the pedestrians'
    positions and velocities and predicts their exact paths (non-yielding
    pedestrians ignore the robot). Each step it rolls out a grid of constant
    actions over `horizon` seconds and scores each by its best estimated arrival
    time if it switched back to heading downhill at some step (before any
    contact). It skips actions that hit a wall within wall_horizon or a pedestrian
    within hard_horizon, and charges contact_cost for a pedestrian hit later in the
    horizon plus pedestrian_weight per second within `comfort` of one (both in
    metres of geodesic, converted to time at full speed). It keeps right: each
    second spent within side_distance of a pedestrian on the wrong side costs
    side_weight. The side is judged in the pedestrian's walking direction, which
    (unlike the robot's heading) doesn't swing round at the junction: the robot
    belongs on an oncoming pedestrian's left and on the right of one walking its
    way, so either passes on the robot's left; crossing ones (more than 45 deg off
    the robot's descent direction) don't count. Without the rule it picked a side
    by tiny cost differences (left 53% of close passes), labels a student policy
    can only average into driving straight. side_weight 0 turns it off. With a
    subgoal lane active (HallwayRaycastEnv subgoal_offset / lane_offset), each
    second spent outside the lane costs lane_weight per metre (beyond
    lane_deadband), except within lane_release of a pedestrian, where it has to
    leave the lane to pass. An off-center lane also replaces keep-right: the
    robot belongs on the lane's side of the pedestrian (left of it for a lane
    left of center), whichever way it walks. An upper-bound reference
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
    speeds = np.empty((n_steps, len(actions)))
    for t in range(n_steps):
        velocity += np.clip(targets - velocity, -max_delta, max_delta)
        pose[:, 0] += velocity[:, 0] * np.cos(pose[:, 2]) * dt
        pose[:, 1] += velocity[:, 0] * np.sin(pose[:, 2]) * dt
        pose[:, 2] += velocity[:, 1] * dt
        poses[t] = pose
        speeds[t] = velocity[:, 0]

    reached = np.hypot(poses[..., 0] - base.goal[0],
                       poses[..., 1] - base.goal[1]) < spec.GOAL_TOLERANCE
    active = np.cumsum(reached, axis=0) == 0  # (T, K): before reaching the goal
    pedestrian_gaps = _pedestrian_gaps(poses, future)  # (T, K, N)
    gaps = pedestrian_gaps.min(axis=2)  # (T, K)
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
    # still needed toward the descent direction + the time lost accelerating back
    # to full speed (without it, stopping ties with driving on and wins by action
    # order, so a robot at rest facing its goal never starts). Each candidate
    # scores its best.
    field = base._goal_field.grid
    row, col = base.map.to_cell(poses[..., 0], poses[..., 1])
    inside = (row >= 0) & (row < base.map.height) & (col >= 0) & (col < base.map.width)
    to_go = np.full(row.shape, np.inf)
    to_go[inside] = field[row[inside], col[inside]]
    to_go = np.where(np.isfinite(to_go), to_go, 1e3)
    heading = _descent_heading(base)
    # Keep right, in each pedestrian's walking frame: unit walking direction, and
    # sense +1 if it walks the robot's way, -1 if oncoming, 0 if crossing.
    walk = np.diff(np.concatenate([base.pedestrians[None], future]), axis=0)  # (T, N, 2)
    walk /= np.maximum(np.linalg.norm(walk, axis=2, keepdims=True), 1e-9)
    along = np.cos(heading) * walk[..., 0] + np.sin(heading) * walk[..., 1]
    sense = np.where(np.abs(along) > np.cos(np.pi / 4), np.sign(along), 0.0)  # (T, N)
    rel_x = poses[..., 0, None] - future[:, None, :, 0]  # robot relative to pedestrian
    rel_y = poses[..., 1, None] - future[:, None, :, 1]
    robot_on_left = walk[:, None, :, 0] * rel_y - walk[:, None, :, 1] * rel_x  # (T, K, N)
    # Oncoming: the robot belongs on its left; walking the robot's way: on its right.
    wrong = sense[:, None, :] * robot_on_left > 0.0
    lane_cost = 0.0
    if base.near_lane is not None:
        _, direction, offset = base.near_lane
        if offset != 0.0:
            # The lane picks the side: the robot belongs left of the pedestrian
            # (across the lane's direction) for a lane left of center.
            normal = np.array([-direction[1], direction[0]])
            across = (poses[..., None, :2] - future[:, None]) @ normal  # (T, K, N)
            wrong = np.sign(offset) * across < 0.0
        released = np.any(pedestrian_gaps < lane_release, axis=2)  # (T, K)
        lane_error = np.abs(base.lane_error(poses[..., :2]))  # (T, K)
        lane_cost = lane_weight * np.sum(
            (active & ~released) * np.maximum(0.0, lane_error - lane_deadband), axis=0) * dt
    wrong_side = np.any(wrong & (pedestrian_gaps < side_distance), axis=2)  # (T, K)
    wrong_side_time = np.sum(active & wrong_side, axis=0) * dt
    turn = np.abs((heading - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi)
    clearance = base._clearance_at(poses[..., 0], poses[..., 1])
    times = (np.arange(n_steps) + 1)[:, None] * dt
    speed_up = (spec.MAX_LINEAR_VEL - speeds) ** 2 \
        / (2 * spec.MAX_LINEAR_ACCEL * spec.MAX_LINEAR_VEL)
    eta = times + to_go / spec.MAX_LINEAR_VEL + turn / spec.MAX_ANGULAR_VEL + speed_up \
        + wall_weight * np.maximum(0.0, wall_clearance - clearance) / spec.MAX_LINEAR_VEL
    eta = np.where(reached, times, eta)
    before_contact = np.arange(n_steps)[:, None] < first
    eta = np.where((active | reached) & before_contact, eta, np.inf)
    cost = eta.min(axis=0) + (pedestrian_weight * discomfort + contact_cost * later_contact
                              + side_weight * wrong_side_time + lane_cost) / spec.MAX_LINEAR_VEL
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
