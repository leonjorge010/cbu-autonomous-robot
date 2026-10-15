"""LiDAR safety layer: overrides velocity commands that would drive into an obstacle.

Runs between the policy and the wheels, on the robot as in simulation. Each
command is checked by simulating the unicycle motion it commands over its
stopping time and testing every scan point against the footprint along the way.
If the robot would come within SAFETY_MARGIN of an obstacle, progressively
gentler commands are tried (less speed, then no turning, then a stop), and the
first safe one is used. Motion that keeps or increases the current gap is always
allowed, so the robot can back away from a wall it is already close to.

ROS-free like spec.py, so the same code runs in the 2D env, in Gazebo and on
the Jetson. SafetyWrapper applies it to a Gymnasium env for evaluation.
"""
from cash_rl import spec
import gymnasium as gym
import numpy as np

# Smallest allowed gap (m) between the footprint and any scan point. The episode's
# collision margin is spec.COLLISION_MARGIN (0.05 m), so this leaves 5 cm to spare.
SAFETY_MARGIN = 0.10
# Fallback commands, as (linear, angular) scale factors, tried in order.
FALLBACKS = ((0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (1.0, 0.0), (0.5, 0.0), (0.25, 0.0),
             (0.0, 1.0), (0.0, 0.5), (0.0, 0.0))
N_HORIZON_SAMPLES = 8


def scan_points(ranges, angles):
    """(N, 2) base_footprint-frame points from sanitized ranges; max-range returns dropped."""
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    keep = ranges < spec.MAX_RANGE
    return np.stack([spec.LIDAR_X + ranges[keep] * np.cos(angles[keep]),
                     ranges[keep] * np.sin(angles[keep])], axis=1)


def _horizon(linear, angular):
    """Seconds to check: one control period plus the time to brake to a stop."""
    return spec.CONTROL_PERIOD + max(abs(linear) / spec.MAX_LINEAR_ACCEL,
                                     abs(angular) / spec.MAX_ANGULAR_ACCEL)


def min_clearance_along(points, linear, angular):
    """Smallest footprint clearance to `points` while holding (linear, angular)."""
    clearance = spec.footprint_clearance(points)
    horizon = _horizon(linear, angular)
    for t in np.linspace(horizon / N_HORIZON_SAMPLES, horizon, N_HORIZON_SAMPLES):
        # Robot pose after t seconds of a constant-twist unicycle motion.
        theta = angular * t
        if abs(angular) > 1e-6:
            x = linear / angular * np.sin(theta)
            y = linear / angular * (1.0 - np.cos(theta))
        else:
            x, y = linear * t, 0.0
        # Obstacle points expressed in that future robot frame.
        c, s = np.cos(theta), np.sin(theta)
        dx, dy = points[:, 0] - x, points[:, 1] - y
        moved = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
        clearance = min(clearance, spec.footprint_clearance(moved))
    return clearance


def limit_command(linear, angular, ranges, angles):
    """Return (linear, angular, overridden) after the safety check.

    ranges must be sanitized (spec.sanitize_ranges); angles are the scan's beam
    angles in the lidar frame.
    """
    points = scan_points(ranges, angles)
    if len(points) == 0:
        return linear, angular, False
    # Don't demand more gap than there is now: moving away from a close wall is fine.
    required = min(SAFETY_MARGIN, spec.footprint_clearance(points)) - 1e-6
    if min_clearance_along(points, linear, angular) >= required:
        return linear, angular, False
    for lin_scale, ang_scale in FALLBACKS:
        candidate = (linear * lin_scale, angular * ang_scale)
        if min_clearance_along(points, *candidate) >= required:
            return candidate[0], candidate[1], True
    return 0.0, 0.0, True


class SafetyWrapper(gym.Wrapper):
    """Applies limit_command to every action; info['safety_override'] marks overridden steps.

    The wrapped env must provide safety_scan() -> (sanitized ranges, angles).
    """

    def step(self, action):
        linear, angular = spec.action_to_cmd(action)
        ranges, angles = self.env.unwrapped.safety_scan()
        linear, angular, overridden = limit_command(linear, angular, ranges, angles)
        if overridden:
            action = spec.cmd_to_action(linear, angular)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['safety_override'] = overridden
        return obs, reward, terminated, truncated, info
