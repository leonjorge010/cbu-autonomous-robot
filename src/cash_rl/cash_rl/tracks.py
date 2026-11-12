"""People seen by comparing a dense LiDAR scan with the map.

The rule-based passing lane (decision.PedestrianLaneDecider) was written against
the simulator's true pedestrian positions. On the robot those positions have to
come from the scan. The policy's 36 beams miss a person about half the time at
4-6 m, which is where the rule commits to a side; a 1-degree scan sees them.
This tracker runs that dense scan, keeps returns that are closer than the map
at the same bearing, and follows those clusters from one step to the next.

ScanDecisionLayer is DecisionLayer with this tracker standing in for
env.pedestrians while the rule picks the lane. The physics pedestrians are
unchanged. evaluate --decision-layer --scan-tracks uses it.
"""
import numpy as np

from cash_rl import spec
from cash_rl.decision import DecisionLayer
from cash_rl.raycast_env import SAFETY_SCAN_ANGLES


# A return this much closer than the map (m) is a dynamic obstacle, not noise
# (the lidar's noise is 1.5 cm) and not the raycaster's half-cell sampling.
MARGIN = 0.4
# Ignore dynamic returns beyond this. The rule only acts inside 6 m.
MAX_RANGE = 8.0
# One noisy beam is not a person. A 0.5 m disc covers several 1-degree beams
# out to MAX_RANGE.
MIN_BEAMS = 2
# World-frame match distance (m). A person moves at most ~0.1 m per step.
ASSOC_DISTANCE = 0.45
# Keep a lost track this many steps so a one-frame miss does not drop the side
# the rule has committed to.
COAST_STEPS = 3


class _Track:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.vel = np.zeros(2)
        self.missed = 0


class ScanTracker:
    """Clusters of scan returns closer than the map, matched across steps."""

    def __init__(self):
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, base):
        """Return (positions, velocities), each (N, 2), in a stable order."""
        detections = _detections(base)
        dt = spec.CONTROL_PERIOD
        matched = np.zeros(len(detections), dtype=bool)
        for track in self.tracks:
            if len(detections):
                distance = np.linalg.norm(detections - track.xy, axis=1)
                distance[matched] = np.inf
                nearest = int(np.argmin(distance))
            else:
                nearest = None
            if nearest is not None and distance[nearest] <= ASSOC_DISTANCE:
                matched[nearest] = True
                new = detections[nearest]
                track.vel = (new - track.xy) / dt
                track.xy = new
                track.missed = 0
            else:
                track.missed += 1
                track.xy = track.xy + track.vel * dt
        for detection, taken in zip(detections, matched):
            if not taken:
                self.tracks.append(_Track(detection))
        self.tracks = [track for track in self.tracks if track.missed <= COAST_STEPS]
        if not self.tracks:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return (np.stack([track.xy for track in self.tracks]),
                np.stack([track.vel for track in self.tracks]))


def _detections(base):
    """World points (K, 2) of dynamic clusters in the latest dense scan."""
    angles = SAFETY_SCAN_ANGLES
    live = base.raycast(angles=angles)
    if base.range_noise:
        live = live.copy()
        finite = np.isfinite(live)
        live[finite] += base.np_random.normal(
            0.0, spec.RANGE_NOISE_STDDEV, int(finite.sum()))
        live[base.np_random.random(len(live)) < spec.BEAM_DROPOUT_PROB] = np.inf
    static = base.raycast(angles=angles, include_pedestrians=False)
    static = np.where(np.isfinite(static), static, spec.MAX_RANGE)
    closer = np.isfinite(live) & (live + MARGIN < static) & (live < MAX_RANGE)
    groups = _clusters(closer)
    if not groups:
        return np.zeros((0, 2))
    x, y, yaw = base.pose
    c, s = np.cos(yaw), np.sin(yaw)
    lidar_x = x + c * spec.LIDAR_X
    lidar_y = y + s * spec.LIDAR_X
    beam_cos = c * np.cos(angles) - s * np.sin(angles)
    beam_sin = s * np.cos(angles) + c * np.sin(angles)
    points = []
    for group in groups:
        px = lidar_x + live[group] * beam_cos[group]
        py = lidar_y + live[group] * beam_sin[group]
        points.append((float(np.mean(px)), float(np.mean(py))))
    return np.asarray(points, dtype=np.float64)


def _clusters(closer):
    """Index groups of consecutive True beams, wrapping around the full scan."""
    index = np.flatnonzero(closer)
    if len(index) == 0:
        return []
    splits = np.where(np.diff(index) > 1)[0] + 1
    groups = np.split(index, splits)
    if (len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(closer) - 1):
        groups = [np.concatenate([groups[-1], groups[0]])] + groups[1:-1]
    return [group for group in groups if len(group) >= MIN_BEAMS]


class ScanDecisionLayer(DecisionLayer):
    """DecisionLayer whose pedestrian positions come from ScanTracker."""

    def __init__(self, env, decider=None, prob=1.0, wait=False):
        super().__init__(env, decider=decider, prob=prob, wait=wait)
        self.tracker = ScanTracker()

    def reset(self, *, seed=None, options=None):
        self.tracker.reset()
        return super().reset(seed=seed, options=options)

    def _decide(self, info):
        base = self.env.unwrapped
        saved_positions = base.pedestrians
        saved_velocities = base._pedestrian_velocities
        positions, velocities = self.tracker.update(base)
        base.pedestrians = positions
        base._pedestrian_velocities = velocities
        try:
            obs, info = super()._decide(info)
        finally:
            base.pedestrians = saved_positions
            base._pedestrian_velocities = saved_velocities
        info['n_tracks'] = len(positions)
        return obs, info
