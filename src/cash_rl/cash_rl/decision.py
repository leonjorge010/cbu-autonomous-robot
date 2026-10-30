"""Rule-based decision layer: moves the subgoal's lane to steer the robot around pedestrians.

The upper half of the hierarchy. The geodesic planner (HallwayRaycastEnv
subgoal_lookahead) picks where the subgoal lies along the route; this picks
which lane it sits in. With no pedestrian in the way the lane is the corridor
center. When one is ahead in the robot's corridor and closing on it (oncoming,
or slower and being overtaken), the lane moves to the side with the most room
beside that pedestrian where the two will meet (keeping right on near-ties),
PASS_SEPARATION from the pedestrian's line as far as the corridor allows, and
stays on that side until the pedestrian is RELEASE_BEHIND behind the robot.
The motion agent only sees the moved subgoal and is paid to follow it
(spec.subgoal_lane_penalty).

Privileged for now: it reads the pedestrians' true positions and velocities
from the 2D env. On the robot those would come from tracking LiDAR clusters.
It only moves the lane sideways; it can't tell the robot to wait, so a
pedestrian walking down the middle of a narrow corridor (the 2.4 m hallway, or
beside the hallway's fountain) leaves no lane wide enough to pass. Nor can it
act within the last subgoal_lookahead of the goal, where lanes are off.

    ros2 run cash_rl evaluate runs/<run>/final_model.zip downhill --decision-layer \
        --random 200 --pedestrians 1 --encounter-prob 1
"""
import gymnasium as gym
import numpy as np

# A pedestrian matters when it is ahead within MAX_AHEAD (m along the route
# direction), within LATERAL_WINDOW (m) of the robot's line, and closing on the
# robot or already within NEAR_AHEAD. Once committed, the side holds until it is
# RELEASE_BEHIND behind the robot or leaves the lateral window.
MAX_AHEAD = 6.0
NEAR_AHEAD = 1.5
LATERAL_WINDOW = 1.5
RELEASE_BEHIND = 0.8
# Lane distance (m) from the pedestrian's line, center to center: the 0.55 m that
# just clears the inflated footprint plus 0.25 m. The env clips the lane to the
# room the corridor has.
PASS_SEPARATION = 0.8
# Room on the two sides within this much (m) of each other counts as a tie: keep right.
SIDE_TIE = 0.15
# Where they will meet is predicted with the robot moving at least
# MIN_PREDICTED_SPEED along the route; the room is the narrowest within
# MEET_SPAN (m) either side of that point.
MIN_PREDICTED_SPEED = 0.3
MEET_SPAN = 1.0


class PedestrianLaneDecider:
    """Pick a lane offset each step from the pedestrians around the robot."""

    def __init__(self, separation=PASS_SEPARATION):
        self.separation = separation
        self.sides = {}  # pedestrian index -> committed side, +1 left / -1 right

    def reset(self):
        self.sides = {}

    def __call__(self, env):
        """Return the lane offset (m, + = left of center) for env's current state."""
        base = env.unwrapped
        if base.near_lane is None or len(base.pedestrians) == 0:
            self.sides = {}
            return 0.0
        _, direction, _ = base.near_lane
        normal = np.array([-direction[1], direction[0]])
        rel = base.pedestrians - base.pose[:2]
        ahead, lateral = rel @ direction, rel @ normal
        robot_speed = base._velocity[0] * float(
            np.cos(base.pose[2]) * direction[0] + np.sin(base.pose[2]) * direction[1])
        closing = robot_speed - base._pedestrian_velocities @ direction
        in_window = (ahead > -RELEASE_BEHIND) & (np.abs(lateral) < LATERAL_WINDOW)
        relevant = in_window & (ahead < MAX_AHEAD) & ((closing > 0.05) | (ahead < NEAR_AHEAD))
        # Drop pedestrians that were passed or left; keep committed ones still in the window.
        self.sides = {i: side for i, side in self.sides.items() if in_window[i]}
        candidates = [i for i in range(len(ahead)) if relevant[i] or i in self.sides]
        if not candidates:
            return 0.0
        # The most urgent: the nearest ahead (passed-but-committed ones count as 0).
        i = min(candidates, key=lambda j: max(ahead[j], 0.0))
        if i not in self.sides:
            speed = max(robot_speed, MIN_PREDICTED_SPEED)
            meet = ahead[i] * speed / max(speed - base._pedestrian_velocities[i] @ direction,
                                          1e-3)
            meet = float(np.clip(meet, 0.0, max(ahead[i], 0.0)))
            self.sides[i] = self._side_with_room(base, base.pose[:2] + meet * direction
                                                 + lateral[i] * normal, direction, normal)
        # Measured from the center of the lane the robot follows (near_lane), so the
        # robot's lane lands PASS_SEPARATION beside the pedestrian's line; the
        # center abreast of the pedestrian jumps at door recesses and open areas.
        point, _, applied = base.near_lane
        center = point - applied * normal
        pedestrian_offset = float((base.pedestrians[i] - center) @ normal)
        return pedestrian_offset + self.sides[i] * self.separation

    @staticmethod
    def _side_with_room(base, meeting, direction, normal):
        """+1 (left) or -1 (right): the side with the wider gap beside the pedestrian.

        meeting is where the pedestrian will be when the robot draws level; each
        side's gap is the narrowest within MEET_SPAN of it.
        """
        gaps = []
        for side in (1.0, -1.0):
            room = np.inf
            for along in np.linspace(-MEET_SPAN, MEET_SPAN, 5):
                point = meeting + along * direction
                # The farthest lane on that side the env allows, abreast of that point.
                lane, _, _ = base._lane_point(point, direction, side * 10.0)
                room = min(room, side * float((lane - point) @ normal))
            gaps.append(room)
        left, right = gaps
        return 1.0 if left > right + SIDE_TIE else -1.0


class DecisionLayer(gym.Wrapper):
    """Runs a decider (default PedestrianLaneDecider) on a HallwayRaycastEnv with lanes.

    Turns subgoal lanes on (lane_offset pinned, not drawn at random) and sets
    lane_offset after every reset and step, rebuilding the observation so the
    policy acts on the new lane. info['decision_offset'] is the requested offset
    (info['lane_offset'] what the corridor allowed). Needs subgoal_lookahead.
    """

    def __init__(self, env, decider=None):
        super().__init__(env)
        if not env.unwrapped.subgoal_lookahead:
            raise ValueError('the decision layer needs subgoal_lookahead')
        self.decider = decider or PedestrianLaneDecider()

    def _decide(self, info):
        base = self.env.unwrapped
        base.lane_offset = float(self.decider(self.env))
        obs = base.observation()  # same scans, subgoal in the new lane
        info['decision_offset'] = base.lane_offset
        info['subgoal'] = base.subgoal.copy()
        if base.lane is not None:
            info['lane_offset'] = base.lane[2]
        return obs, info

    def reset(self, *, seed=None, options=None):
        self.decider.reset()
        options = dict(options or {}, lane_offset=0.0)
        _, info = self.env.reset(seed=seed, options=options)
        return self._decide(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            obs, info = self._decide(info)
        return obs, reward, terminated, truncated, info


def passes_on_decided_side(env, decided_side):
    """Return True if the robot is on decided_side (+1 left / -1 right) of the nearest pedestrian.

    Measured across the near lane's direction; for tests and evaluation.
    """
    base = env.unwrapped
    if base.near_lane is None or len(base.pedestrians) == 0:
        return True
    _, direction, _ = base.near_lane
    normal = np.array([-direction[1], direction[0]])
    i = int(np.argmin(np.hypot(*(base.pedestrians - base.pose[:2]).T)))
    return bool(decided_side * float((base.pose[:2] - base.pedestrians[i]) @ normal) > 0)
