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
It only moves the lane sideways. While someone is closing inside
URGENT_DISTANCE the subgoal and the followed lane are pulled in to
URGENT_LOOKAHEAD, and a lane is kept all the way to the goal so the last
metres can still pass. With wait=True, if the chosen lane still leaves a
person ahead inside the robot's body, the layer holds the command at zero.
A person walking off the lane keeps that hold. A stopped person with an open
side is driven past at full speed; with no open side the robot backs up until
one appears, and holds if a wall is behind it. The passing side is chosen once
and held until that person is behind the robot.

    ros2 run cash_rl evaluate runs/<run>/final_model.zip downhill --decision-layer \
        --random 200 --pedestrians 1 --encounter-prob 1
"""
import gymnasium as gym
import numpy as np

from cash_rl import spec

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
# A person this close to the lane the robot will actually drive (m, center to
# center) still overlaps the inflated footprint. Half the footprint plus the
# 0.25 m pedestrian disc.
PASS_CLEARANCE = 0.545
# Inside this distance (m) a closing person is a pass that has to happen now:
# the subgoal and the lane the teacher follows are pulled in to URGENT_LOOKAHEAD.
URGENT_DISTANCE = 3.0
URGENT_LOOKAHEAD = 0.5
# Hold still this long (s) before giving up on a stop. A person walking off the
# lane keeps the hold. A stopped person is then driven past, or backed away from.
ESCAPE_WAIT = 0.3
# Below this speed (m/s) a person counts as stopped. Yielders keep their walking
# velocity stored while they stand still, so this is measured from the last step.
STOPPED_SPEED = 0.05
# Creep forward at this fraction of full speed while a person is still moving
# and the open side already clears the body. A stopped person is passed at
# PASS_ACTION until they are PASS_SEPARATION off to the side, then at full speed.
CREEP_ACTION = 0.35
PASS_ACTION = 0.6
DRIVE_ACTION = 1.0
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
        # Drop pedestrians that were passed, left the window, or disappeared.
        # A lost track shortens the list; its old index must not be read.
        n = len(ahead)
        self.sides = {i: side for i, side in self.sides.items()
                      if i < n and in_window[i]}
        candidates = [i for i in range(len(ahead)) if relevant[i] or i in self.sides]
        if not candidates:
            return 0.0
        # The most urgent: the nearest ahead (passed-but-committed ones count as 0).
        i = min(candidates, key=lambda j: max(ahead[j], 0.0))
        # The side is chosen once and held for the whole pass. Re-picking it while
        # someone crosses flips the lane into them.
        if i not in self.sides:
            velocity = base._pedestrian_velocities[i]
            speed = max(robot_speed, MIN_PREDICTED_SPEED)
            meet = ahead[i] * speed / max(speed - velocity @ direction, 1e-3)
            meet = float(np.clip(meet, 0.0, max(ahead[i], 0.0)))
            meeting = base.pose[:2] + meet * direction + lateral[i] * normal
            self.sides[i] = self._side_with_room(base, meeting, direction, normal)
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
    prob < 1 (for training) runs it only in that share of episodes, drawn per reset;
    the others pass through untouched (the env's own lanes, random if subgoal_offset).
    """

    def __init__(self, env, decider=None, prob=1.0, wait=False):
        super().__init__(env)
        if not env.unwrapped.subgoal_lookahead:
            raise ValueError('the decision layer needs subgoal_lookahead')
        self.decider = decider or PedestrianLaneDecider()
        self.prob = prob
        self.wait = wait
        self.active = True
        self._waiting = False
        self._wait_time = 0.0
        self._prev_pedestrians = None
        self._rng = np.random.default_rng()

    def _route_frame(self, base):
        """(ahead distances, unit route direction, unit left normal) of each person."""
        if base.near_lane is not None:
            _, direction, _ = base.near_lane
        else:
            direction = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
        direction = np.asarray(direction, dtype=np.float64)
        normal = np.array([-direction[1], direction[0]])
        rel = base.pedestrians - base.pose[:2]
        return rel @ direction, direction, normal

    def _someone_ahead(self, base):
        """True when a person is still in front of the robot, inside the rule's window."""
        if len(base.pedestrians) == 0:
            return False
        ahead, _, _ = self._route_frame(base)
        return bool(np.any((ahead > -RELEASE_BEHIND) & (ahead < MAX_AHEAD)))

    def _urgent(self, base):
        """True when a person close ahead is still in the corridor the robot is using."""
        if len(base.pedestrians) == 0 or base.near_lane is None:
            return False
        ahead, direction, normal = self._route_frame(base)
        lateral = (base.pedestrians - base.pose[:2]) @ normal
        robot_speed = base._velocity[0] * float(
            np.cos(base.pose[2]) * direction[0] + np.sin(base.pose[2]) * direction[1])
        closing = robot_speed - base._pedestrian_velocities @ direction
        close = (ahead > 0.0) & (ahead < URGENT_DISTANCE) & (np.abs(lateral) < LATERAL_WINDOW)
        return bool(np.any(close & ((closing > 0.05) | (ahead < NEAR_AHEAD))))

    def _prepare_lane(self, base):
        """Pull the subgoal in, and keep a lane, while a person is still ahead.

        Called before the decider when the last metres have no lane yet, and
        again after the offset is chosen so the observation matches it.
        """
        if not hasattr(base, 'pass_lookahead'):
            return
        ahead = self._someone_ahead(base)
        base.lane_to_goal = ahead
        if ahead and base.near_lane is None:
            base.observation()
        base.pass_lookahead = URGENT_LOOKAHEAD if self._urgent(base) else None

    def _blocking_index(self):
        """Index of the nearest person the current lane would still hit, or None."""
        base = self.env.unwrapped
        if len(base.pedestrians) == 0:
            return None
        if base.near_lane is not None:
            point, direction, applied = base.near_lane
            normal = np.array([-direction[1], direction[0]])
            center = point - applied * normal
            rel = base.pedestrians - base.pose[:2]
            ahead = rel @ direction
            gap = np.abs((base.pedestrians - center) @ normal - applied)
            blocked = np.flatnonzero((ahead > -0.3) & (ahead < MAX_AHEAD) & (gap < PASS_CLEARANCE))
        else:
            gaps = base.pedestrian_gaps()
            forward = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
            ahead = (base.pedestrians - base.pose[:2]) @ forward
            blocked = np.flatnonzero((ahead > 0.0) & (ahead < 2.5) & (gaps < 0.35))
        if len(blocked) == 0:
            return None
        return int(blocked[np.argmin(ahead[blocked])])

    def _pass_blocked(self):
        """True when someone ahead still overlaps the lane the robot would drive.

        With no lane (the last metres before the goal) it is someone just in
        front of the footprint. Checked again every step, so the hold ends once
        they have passed or a clear lane appears.
        """
        return self._blocking_index() is not None

    def _clearing(self, index):
        """True when that person is walking off the lane faster than along it."""
        base = self.env.unwrapped
        if base.near_lane is None:
            return False
        point, direction, applied = base.near_lane
        normal = np.array([-direction[1], direction[0]])
        center = point - applied * normal
        signed = float((base.pedestrians[index] - center) @ normal - applied)
        velocity = base._pedestrian_velocities[index]
        lateral = float(velocity @ normal)
        along = abs(float(velocity @ direction))
        return abs(lateral) > 0.3 and abs(lateral) > along and lateral * np.sign(signed) > 0.0

    def _remember_pedestrians(self):
        base = self.env.unwrapped
        self._prev_pedestrians = (np.array(base.pedestrians, copy=True)
                                  if len(base.pedestrians) else np.zeros((0, 2)))

    def _stopped(self, index):
        """True when that person barely moved on the last step.

        A yielder who has stopped still stores a walking velocity, so the stored
        value is only the fallback for the first step, before a displacement exists.
        """
        base = self.env.unwrapped
        prev = self._prev_pedestrians
        if prev is None or len(prev) != len(base.pedestrians):
            return float(np.linalg.norm(base._pedestrian_velocities[index])) < STOPPED_SPEED
        moved = float(np.linalg.norm(base.pedestrians[index] - prev[index]))
        return moved / spec.CONTROL_PERIOD < STOPPED_SPEED

    def _rear_open(self, base):
        """True when a short step backward still clears the walls."""
        heading = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
        back = base.pose[:2] - heading * (spec.FOOTPRINT_HALF_LENGTH + 0.15)
        return float(base._clearance_at(back[0], back[1])) >= spec.COLLISION_MARGIN

    def _backup(self, base, turn):
        """Reverse while the way behind is open. Hold still against a wall."""
        if not self._rear_open(base):
            return np.zeros(2, dtype=np.float32)
        return np.array([-1.0, float(np.clip(-0.5 * turn, -1.0, 1.0))], dtype=np.float32)

    def _escape_action(self):
        """Drive past a stopped person, or back up until a gap opens.

        None means keep holding still: the person is walking off the lane.
        A moving person with room is still crept past. There is no timer that
        gives up and drives into a gap that is still too narrow.
        """
        base = self.env.unwrapped
        index = self._blocking_index()
        if index is None or self._clearing(index):
            return None
        if base.near_lane is None:
            return self._backup(base, 0.0)
        _, direction, _ = base.near_lane
        normal = np.array([-direction[1], direction[0]])
        side = self.decider.sides.get(index)
        if side is None:
            side = self.decider._side_with_room(
                base, base.pedestrians[index], direction, normal)
        lane, _, _ = base._lane_point(base.pedestrians[index], direction, side * 10.0)
        room = side * float((lane - base.pedestrians[index]) @ normal)
        heading = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
        facing = float(heading @ direction)
        turn = float(side * (1.0 if facing >= 0.0 else -1.0))
        turn = float(np.clip(turn, -1.0, 1.0))
        if room >= PASS_CLEARANCE:
            if self._stopped(index):
                return self._pass_stopped(base, index, turn)
            return np.array([CREEP_ACTION, turn], dtype=np.float32)
        return self._backup(base, turn)

    def _pass_stopped(self, base, index, turn):
        """Turn clear of a stopped person, then pass, and only then go full speed.

        Full speed while they are still near the nose is what turned the mix's
        timeouts into crashes. PASS_ACTION is the speed in between.
        """
        heading = np.array([np.cos(base.pose[2]), np.sin(base.pose[2])])
        rel = base.pedestrians[index] - base.pose[:2]
        along = float(rel @ heading)
        lateral = abs(float(rel @ np.array([-heading[1], heading[0]])))
        if along > 0.0 and lateral < PASS_CLEARANCE:
            if along < spec.PEDESTRIAN_COMFORT_DISTANCE:
                return self._backup(base, turn)
            return np.array([0.0, turn], dtype=np.float32)
        if along > -RELEASE_BEHIND and lateral < PASS_SEPARATION:
            return np.array([PASS_ACTION, turn], dtype=np.float32)
        return np.array([DRIVE_ACTION, turn], dtype=np.float32)

    def _decide(self, info):
        base = self.env.unwrapped
        self._prepare_lane(base)
        base.lane_offset = float(self.decider(self.env))
        self._prepare_lane(base)
        obs = base.observation()  # same scans, subgoal in the new lane
        self._waiting = bool(self.wait and self._pass_blocked())
        if not self._waiting:
            self._wait_time = 0.0
        info['decision_offset'] = base.lane_offset
        info['decision_wait'] = self._waiting
        info['subgoal'] = base.subgoal.copy()
        if base.lane is not None:
            info['lane_offset'] = base.lane[2]
        return obs, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.active = self.prob >= 1 or self._rng.random() < self.prob
        self._waiting = False
        self._wait_time = 0.0
        self._prev_pedestrians = None
        base = self.env.unwrapped
        if hasattr(base, 'pass_lookahead'):
            base.pass_lookahead = None
            base.lane_to_goal = False
        if not self.active:
            return self.env.reset(seed=seed, options=options)
        self.decider.reset()
        options = dict(options or {}, lane_offset=0.0)
        _, info = self.env.reset(seed=seed, options=options)
        self._remember_pedestrians()
        return self._decide(info)

    def step(self, action):
        escape = None
        if self.active and self._waiting:
            self._wait_time += spec.CONTROL_PERIOD
            if self._wait_time >= ESCAPE_WAIT:
                escape = self._escape_action()
        waited = self.active and self._waiting and escape is None
        if waited:
            action = np.zeros(2, dtype=np.float32)
        elif escape is not None:
            action = escape
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._remember_pedestrians()
        info['waited'] = waited
        info['escaped'] = escape is not None
        if self.active and not (terminated or truncated):
            obs, info = self._decide(info)
            info['waited'] = waited
            info['escaped'] = escape is not None
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
