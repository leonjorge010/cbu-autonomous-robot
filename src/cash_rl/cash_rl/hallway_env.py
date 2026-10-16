"""Gymnasium environment wrapping the hallway sim over ROS 2 topics/services.

Requires `sim.launch.py` (cash_sim) plus this package's bridge for the
world-control/set-pose services and the robot's ground-truth pose already
running — see launch/train_env.launch.py.
"""
import time

from cash_rl import spec
from cash_rl.occupancy_map import OccupancyMap
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity, WorldControl, WorldReset
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from sensor_msgs.msg import LaserScan

WORLD_NAME = 'hallway'
ROBOT_NAME = 'cash_bot'
# sim.launch.py's hardcoded spawn z.
SPAWN_Z = 0.02
SCAN_WAIT_TIMEOUT = 5.0
# After a teleport, wait until the ground-truth pose is this close to the request.
TELEPORT_TOLERANCE = 0.1


class HallwayEnv(gym.Env):
    """Scan-in / cmd_vel-out environment for a single cash_bot in hallway.sdf.

    Observation, action, collision and reward definitions live in cash_rl.spec,
    shared with HallwayRaycastEnv. reset(options=...) accepts 'spawn' (x, y, yaw),
    default spec.SPAWN_POSE, and 'goal' (x, y), default a random spec.DOOR_GOALS
    entry. Robot pose comes from Gazebo ground truth (/model/cash_bot/pose), and
    geodesic goal distances for the reward from the SLAM map (map_yaml).
    """

    metadata = {'render_modes': []}

    def __init__(self, map_yaml=None):
        super().__init__()
        self._map = OccupancyMap(map_yaml)
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = rclpy.create_node('cash_rl_env')
        self._latest_scan = None
        self._latest_pose = None
        self._node.create_subscription(
            LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self._node.create_subscription(
            PoseStamped, f'/model/{ROBOT_NAME}/pose', self._on_pose, 10)
        self._cmd_pub = self._node.create_publisher(Twist, '/cmd_vel', 10)
        self._reset_client = self._node.create_client(
            ControlWorld, f'/world/{WORLD_NAME}/control')
        self._set_pose_client = self._node.create_client(
            SetEntityPose, f'/world/{WORLD_NAME}/set_pose')

        self.observation_space = spec.observation_space()
        self.action_space = spec.action_space()

        self.goal = np.zeros(2)
        self._goal_field = None
        self._goal_distance = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._scan_stack = None
        self._step_count = 0
        self._max_steps = 0

    def _on_scan(self, msg):
        self._latest_scan = msg

    def _on_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._latest_pose = np.array([p.x, p.y, yaw])

    def _spin_for(self, duration_s):
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=max(0.0, deadline - time.monotonic()))

    def _wait_for_scan(self):
        self._latest_scan = None
        deadline = time.monotonic() + SCAN_WAIT_TIMEOUT
        while self._latest_scan is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    'No /scan message received — is sim.launch.py running?')
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def _wait_for_pose_near(self, x, y):
        deadline = time.monotonic() + SCAN_WAIT_TIMEOUT
        while (self._latest_pose is None
               or np.hypot(self._latest_pose[0] - x, self._latest_pose[1] - y)
               > TELEPORT_TOLERANCE):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f'Ground-truth pose not near ({x}, {y}) — is /model/{ROBOT_NAME}/pose '
                    'bridged (launch/train_env.launch.py)?')
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def _sanitized_ranges(self):
        return spec.sanitize_ranges(self._latest_scan.ranges)

    def _scan_angles(self):
        scan = self._latest_scan
        return scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment

    def safety_scan(self):
        """(ranges, angles) of the latest full scan for safety.limit_command."""
        return self._sanitized_ranges(), self._scan_angles()

    def _observation(self, new_episode=False):
        """Push the latest scan onto the stack (or start one) and build the observation."""
        obs_ranges = self._sanitized_ranges()[spec.OBS_BEAM_INDICES]
        self._scan_stack = spec.new_scan_stack(obs_ranges) if new_episode \
            else spec.push_scan(self._scan_stack, obs_ranges)
        return spec.build_observation(
            self._scan_stack, self._latest_pose, self.goal, self._prev_action)

    def _info(self, **extra):
        return {'pose': self._latest_pose.copy(), 'goal': self.goal.copy(),
                'goal_distance': self._goal_distance, **extra}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        spawn = options.get('spawn', spec.SPAWN_POSE)
        goal = options.get('goal')
        if goal is None:
            names = sorted(spec.DOOR_GOALS)
            goal = spec.DOOR_GOALS[names[self.np_random.integers(len(names))]]
        self._cmd_pub.publish(Twist())

        if not self._reset_client.wait_for_service(timeout_sec=SCAN_WAIT_TIMEOUT):
            raise RuntimeError(
                f'/world/{WORLD_NAME}/control service unavailable — is the '
                'ControlWorld service bridge running? (launch/train_env.launch.py)')
        control_request = ControlWorld.Request()
        control_request.world_control = WorldControl(reset=WorldReset(model_only=True))
        future = self._reset_client.call_async(control_request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=SCAN_WAIT_TIMEOUT)

        # ControlWorld's model_only reset doesn't restore pose for entities spawned
        # after world load (like cash_bot), so teleport it to the spawn explicitly.
        if not self._set_pose_client.wait_for_service(timeout_sec=SCAN_WAIT_TIMEOUT):
            raise RuntimeError(
                f'/world/{WORLD_NAME}/set_pose service unavailable — is the '
                'SetEntityPose service bridge running? (launch/train_env.launch.py)')
        pose_request = SetEntityPose.Request()
        pose_request.entity = Entity(name=ROBOT_NAME, type=Entity.MODEL)
        x, y, yaw = spawn
        pose_request.pose = Pose(
            position=Point(x=float(x), y=float(y), z=SPAWN_Z),
            orientation=Quaternion(z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2))))
        future = self._set_pose_client.call_async(pose_request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=SCAN_WAIT_TIMEOUT)

        self._wait_for_pose_near(x, y)
        self._wait_for_scan()
        self.goal = np.array(goal, dtype=np.float64)
        self._goal_field = self._map.distance_field(*self.goal)
        self._goal_distance = self._goal_field(*self._latest_pose[:2])
        if not np.isfinite(self._goal_distance):
            raise RuntimeError(f'Goal {self.goal} unreachable from {self._latest_pose}')
        self._prev_action[:] = 0.0
        self._step_count = 0
        self._max_steps = spec.episode_step_limit(self._goal_distance)
        return self._observation(new_episode=True), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        linear, angular = spec.action_to_cmd(action)
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self._cmd_pub.publish(twist)

        self._spin_for(spec.CONTROL_PERIOD)
        self._step_count += 1
        self._prev_action = action

        ranges = self._sanitized_ranges()
        angles = self._scan_angles()
        collided = spec.scan_hits_footprint(ranges, angles)
        clearance = spec.scan_footprint_clearance(ranges, angles)
        goal_distance = self._goal_field(*self._latest_pose[:2])
        reward, reached = spec.step_outcome(
            self._goal_distance, goal_distance, self._latest_pose, self.goal, collided,
            clearance)
        if np.isfinite(goal_distance):
            self._goal_distance = goal_distance
        obs = self._observation()
        terminated = collided or reached
        truncated = self._step_count >= self._max_steps
        info = self._info(min_range=float(np.min(ranges)), clearance=clearance,
                          collided=collided, is_success=reached)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._cmd_pub.publish(Twist())
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
