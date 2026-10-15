"""Robot/sensor spec and episode rules shared by every C.A.$.H training env.

HallwayEnv (Gazebo over ROS) and HallwayRaycastEnv (fast 2D map raycaster) both
take their observation, action, collision and reward definitions from here, so a
policy trained in one drops into the other unchanged. No ROS imports allowed:
the 2D env must run without a ROS install.

Values mirror cash_sim/urdf/cash_bot.urdf.xacro — keep them in sync.
"""
import gymnasium as gym
import numpy as np

# gpu_lidar: 360 samples over [-pi, pi] inclusive, so increment = 2*pi/359.
N_SCAN_BEAMS = 360
SCAN_ANGLE_MIN = -3.14159
SCAN_ANGLE_MAX = 3.14159
SCAN_ANGLE_INCREMENT = (SCAN_ANGLE_MAX - SCAN_ANGLE_MIN) / (N_SCAN_BEAMS - 1)
MIN_RANGE = 0.15
MAX_RANGE = 12.0
RANGE_NOISE_STDDEV = 0.01
LIDAR_X = 0.05  # lidar_link offset ahead of base_footprint

N_OBS_BEAMS = 36
OBS_BEAM_INDICES = np.arange(0, N_SCAN_BEAMS, N_SCAN_BEAMS // N_OBS_BEAMS)[:N_OBS_BEAMS]
OBS_BEAM_ANGLES = SCAN_ANGLE_MIN + OBS_BEAM_INDICES * SCAN_ANGLE_INCREMENT
SCAN_BEAM_ANGLES = SCAN_ANGLE_MIN + np.arange(N_SCAN_BEAMS) * SCAN_ANGLE_INCREMENT

# Footprint in base_footprint frame: 0.50 m chassis length; width is set by the
# wheels' outer faces (wheel_sep 0.45 + wheel_width 0.04), not the 0.40 m chassis.
FOOTPRINT_HALF_LENGTH = 0.25
FOOTPRINT_HALF_WIDTH = 0.245
# An obstacle this close to the footprint counts as a collision and ends the episode.
COLLISION_MARGIN = 0.05

MAX_LINEAR_VEL = 0.5
# Reverse is capped well below forward speed: the progress reward pays the same for
# driving backward, so with a symmetric limit policies reversed ~half the time
# instead of turning around. Slow reverse still lets the robot back off a wall.
MAX_REVERSE_VEL = 0.15
MAX_ANGULAR_VEL = 1.0
# DiffDrive plugin limits (max_linear_acceleration / max_angular_acceleration).
MAX_LINEAR_ACCEL = 1.0
MAX_ANGULAR_ACCEL = 2.0

CONTROL_PERIOD = 0.1
# Episode time limit: EPISODE_TIME_FACTOR x the time to cover the spawn's geodesic
# goal distance at MAX_LINEAR_VEL, plus EPISODE_TIME_SLACK seconds for turning.
EPISODE_TIME_FACTOR = 2.0
EPISODE_TIME_SLACK = 10.0

# Goal-reaching task. Observation = [N_OBS_BEAMS ranges / MAX_RANGE,
# goal distance / MAX_GOAL_DISTANCE, cos(goal bearing), sin(goal bearing),
# previous action (2)], with the goal in the robot's base_footprint frame.
GOAL_TOLERANCE = 0.3
MAX_GOAL_DISTANCE = 30.0
N_OBS = N_OBS_BEAMS + 3 + 2

# Reward per step: PROGRESS_WEIGHT * (decrease in geodesic goal distance, m)
# - TIME_PENALTY, plus GOAL_REWARD on reaching the goal or - COLLISION_PENALTY
# on collision (both end the episode). Progress is potential-based shaping, so
# its episode total depends only on start and end distance, not the path.
PROGRESS_WEIGHT = 10.0
TIME_PENALTY = 0.01
GOAL_REWARD = 10.0
COLLISION_PENALTY = 20.0

# Fixed evaluation goals: the corridor centerline in front of each door in
# cash_sim/worlds/hallway.sdf (world frame == map frame).
DOOR_GOALS = {
    'door_101': (4.0, 0.0),
    'door_105': (7.0, 0.0),
    'door_102': (9.0, 0.0),
    'door_103': (20.0, 0.0),
    'door_104': (24.0, 0.0),
    'door_201': (14.0, 6.0),
    'door_202': (14.0, 9.0),
}
SPAWN_POSE = (0.0, 0.0, 0.0)  # sim.launch.py's default x, y, yaw


def observation_space():
    low = np.concatenate([np.zeros(N_OBS_BEAMS), [0.0, -1.0, -1.0], [-1.0, -1.0]])
    high = np.ones(N_OBS)
    return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32),
                          dtype=np.float32)


def action_space():
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def action_to_cmd(action):
    """Map a [-1, 1]^2 action to (linear_x m/s, angular_z rad/s).

    action[0] in [0, 1] scales MAX_LINEAR_VEL and in [-1, 0) MAX_REVERSE_VEL.
    """
    action = np.clip(action, -1.0, 1.0)
    linear = float(action[0]) * (MAX_LINEAR_VEL if action[0] >= 0 else MAX_REVERSE_VEL)
    return linear, float(action[1]) * MAX_ANGULAR_VEL


def sanitize_ranges(ranges):
    """NaN/+inf (no return) -> MAX_RANGE; -inf (below range_min) -> 0."""
    ranges = np.asarray(ranges, dtype=np.float32)
    ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0)
    return np.clip(ranges, 0.0, MAX_RANGE)


def goal_in_robot_frame(pose, goal):
    """Return (distance, bearing) of goal (x, y) from pose (x, y, yaw)."""
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    return float(np.hypot(dx, dy)), float(np.arctan2(dy, dx) - pose[2])


def build_observation(obs_ranges, pose, goal, prev_action):
    """Assemble the N_OBS observation from sanitized N_OBS_BEAMS ranges."""
    distance, bearing = goal_in_robot_frame(pose, goal)
    return np.concatenate([
        np.asarray(obs_ranges, dtype=np.float32) / MAX_RANGE,
        [min(distance / MAX_GOAL_DISTANCE, 1.0), np.cos(bearing), np.sin(bearing)],
        np.clip(prev_action, -1.0, 1.0),
    ]).astype(np.float32)


def scan_hits_footprint(ranges, angles):
    """Return True if any scan return lies inside the margin-inflated robot footprint."""
    x = LIDAR_X + ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return bool(np.any(
        (np.abs(x) < FOOTPRINT_HALF_LENGTH + COLLISION_MARGIN)
        & (np.abs(y) < FOOTPRINT_HALF_WIDTH + COLLISION_MARGIN)))


def episode_step_limit(goal_distance):
    """Return the step budget for an episode starting goal_distance (geodesic) away."""
    seconds = EPISODE_TIME_FACTOR * goal_distance / MAX_LINEAR_VEL + EPISODE_TIME_SLACK
    return int(np.ceil(seconds / CONTROL_PERIOD))


def step_outcome(prev_goal_distance, goal_distance, pose, goal, collided):
    """Return (reward, reached) for one step.

    goal_distance/prev_goal_distance are geodesic (around walls); NaN (pose off
    the traversable map) counts as no progress. Reaching is Euclidean, within
    GOAL_TOLERANCE, and doesn't count on a colliding step.
    """
    progress = goal_distance - prev_goal_distance
    progress = 0.0 if not np.isfinite(progress) else -progress
    reached = (not collided) and goal_in_robot_frame(pose, goal)[0] < GOAL_TOLERANCE
    reward = PROGRESS_WEIGHT * progress - TIME_PENALTY
    if reached:
        reward += GOAL_REWARD
    if collided:
        reward -= COLLISION_PENALTY
    return reward, reached
