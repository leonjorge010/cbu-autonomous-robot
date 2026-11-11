from cash_rl import safety, spec
from cash_rl.raycast_env import HallwayRaycastEnv
import numpy as np
import pytest

HL, HW = spec.FOOTPRINT_HALF_LENGTH, spec.FOOTPRINT_HALF_WIDTH


def wall_scan(x=None, y=None):
    """Sanitized 1-degree scan of an infinite wall at base_footprint x = `x` or y = `y`."""
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    with np.errstate(divide='ignore'):
        if x is not None:  # lidar sits LIDAR_X ahead of base_footprint
            ranges = np.where(np.sign(c) == np.sign(x - spec.LIDAR_X),
                              (x - spec.LIDAR_X) / c, np.inf)
        else:
            ranges = np.where(np.sign(s) == np.sign(y), y / s, np.inf)
    return spec.sanitize_ranges(ranges), angles


def test_open_space_passes_commands_through():
    ranges, angles = wall_scan(x=8.0)
    assert safety.limit_command(0.5, 0.7, ranges, angles) == (0.5, 0.7, False)


def test_full_speed_toward_a_close_wall_is_slowed():
    ranges, angles = wall_scan(x=HL + 0.25)  # 25 cm from the front edge
    linear, angular, overridden = safety.limit_command(0.5, 0.0, ranges, angles)
    assert overridden and 0.0 <= linear < 0.5
    assert safety.min_clearance_along(safety.scan_points(ranges, angles), linear, 0.0) \
        >= safety.SAFETY_MARGIN - 1e-6


def test_wall_behind_does_not_limit_driving_forward():
    ranges, angles = wall_scan(x=-HL - 0.12)
    assert safety.limit_command(0.5, 0.0, ranges, angles) == (0.5, 0.0, False)


def test_inside_the_margin_backing_away_is_allowed_but_forward_is_not():
    ranges, angles = wall_scan(x=HL + 0.07)  # front wall already closer than SAFETY_MARGIN
    assert safety.limit_command(-0.15, 0.0, ranges, angles) == (-0.15, 0.0, False)
    linear, _, overridden = safety.limit_command(0.5, 0.0, ranges, angles)
    assert overridden and linear == 0.0


def test_turning_beside_a_close_wall_is_blocked():
    # Rotating in place swings a front or rear corner into a wall this close.
    ranges, angles = wall_scan(y=HW + 0.07)
    for angular in (1.0, -1.0):
        assert safety.limit_command(0.0, angular, ranges, angles)[1] == 0.0


@pytest.mark.parametrize('yaw', [0.0, 0.6])
def test_wrapped_env_never_hits_the_end_wall(yaw):
    env = safety.SafetyWrapper(HallwayRaycastEnv(range_noise=False))
    env.reset(seed=0, options={'spawn': (25.0, 0.0, yaw), 'goal': spec.DOOR_GOALS['door_101']})
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([1.0, 0.0]))
        assert not info['collided']
        if terminated:
            break
    assert info['safety_override']
