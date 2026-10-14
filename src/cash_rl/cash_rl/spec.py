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
MAX_ANGULAR_VEL = 1.0
# DiffDrive plugin limits (max_linear_acceleration / max_angular_acceleration).
MAX_LINEAR_ACCEL = 1.0
MAX_ANGULAR_ACCEL = 2.0

CONTROL_PERIOD = 0.1
MAX_EPISODE_STEPS = 500
COLLISION_PENALTY = 10.0


def observation_space():
    return gym.spaces.Box(low=0.0, high=1.0, shape=(N_OBS_BEAMS,), dtype=np.float32)


def action_space():
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def action_to_cmd(action):
    """Map a [-1, 1]^2 action to (linear_x m/s, angular_z rad/s)."""
    action = np.clip(action, -1.0, 1.0)
    return float(action[0]) * MAX_LINEAR_VEL, float(action[1]) * MAX_ANGULAR_VEL


def sanitize_ranges(ranges):
    """NaN/+inf (no return) -> MAX_RANGE; -inf (below range_min) -> 0."""
    ranges = np.asarray(ranges, dtype=np.float32)
    ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0)
    return np.clip(ranges, 0.0, MAX_RANGE)


def observation_from_ranges(obs_ranges):
    """Normalize the N_OBS_BEAMS ranges (already sanitized) to [0, 1]."""
    return (np.asarray(obs_ranges, dtype=np.float32) / MAX_RANGE).astype(np.float32)


def scan_hits_footprint(ranges, angles):
    """Return True if any scan return lies inside the margin-inflated robot footprint."""
    x = LIDAR_X + ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return bool(np.any(
        (np.abs(x) < FOOTPRINT_HALF_LENGTH + COLLISION_MARGIN)
        & (np.abs(y) < FOOTPRINT_HALF_WIDTH + COLLISION_MARGIN)))


def placeholder_reward(linear_cmd, collided):
    """Forward progress minus a collision penalty. Replace once goals exist."""
    reward = linear_cmd * CONTROL_PERIOD
    if collided:
        reward -= COLLISION_PENALTY
    return reward
