"""Gymnasium environment wrapping the C.A.$.H hallway sim over ROS 2 topics/services.

Requires `sim.launch.py` (cash_sim) plus this package's service bridge for
/world/hallway/control already running — see launch/train_env.launch.py.
"""
import time

from cash_rl import spec
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity, WorldControl, WorldReset
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from sensor_msgs.msg import LaserScan

WORLD_NAME = 'hallway'
ROBOT_NAME = 'cash_bot'
# Must match sim.launch.py's x/y/yaw defaults (0.0) and its hardcoded spawn z (0.02).
SPAWN_POSE = Pose(position=Point(x=0.0, y=0.0, z=0.02), orientation=Quaternion(w=1.0))
SCAN_WAIT_TIMEOUT = 5.0


class HallwayEnv(gym.Env):
    """Scan-in / cmd_vel-out environment for a single cash_bot in hallway.sdf.

    Observation, action, collision and reward definitions live in cash_rl.spec,
    shared with HallwayRaycastEnv.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = rclpy.create_node('cash_rl_env')
        self._latest_scan = None
        self._node.create_subscription(
            LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self._cmd_pub = self._node.create_publisher(Twist, '/cmd_vel', 10)
        self._reset_client = self._node.create_client(
            ControlWorld, f'/world/{WORLD_NAME}/control')
        self._set_pose_client = self._node.create_client(
            SetEntityPose, f'/world/{WORLD_NAME}/set_pose')

        self.observation_space = spec.observation_space()
        self.action_space = spec.action_space()

        self._step_count = 0

    def _on_scan(self, msg):
        self._latest_scan = msg

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

    def _sanitized_ranges(self):
        return spec.sanitize_ranges(self._latest_scan.ranges)

    def _scan_angles(self):
        scan = self._latest_scan
        return scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment

    def _observation(self):
        return spec.observation_from_ranges(self._sanitized_ranges()[spec.OBS_BEAM_INDICES])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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
        # after world load (like cash_bot), so teleport it back to spawn explicitly.
        if not self._set_pose_client.wait_for_service(timeout_sec=SCAN_WAIT_TIMEOUT):
            raise RuntimeError(
                f'/world/{WORLD_NAME}/set_pose service unavailable — is the '
                'SetEntityPose service bridge running? (launch/train_env.launch.py)')
        pose_request = SetEntityPose.Request()
        pose_request.entity = Entity(name=ROBOT_NAME, type=Entity.MODEL)
        pose_request.pose = SPAWN_POSE
        future = self._set_pose_client.call_async(pose_request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=SCAN_WAIT_TIMEOUT)

        self._wait_for_scan()
        self._step_count = 0
        return self._observation(), {}

    def step(self, action):
        linear, angular = spec.action_to_cmd(action)
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self._cmd_pub.publish(twist)

        self._spin_for(spec.CONTROL_PERIOD)
        self._step_count += 1

        obs = self._observation()
        ranges = self._sanitized_ranges()
        min_range = float(np.min(ranges))
        collided = spec.scan_hits_footprint(ranges, self._scan_angles())
        terminated = collided
        truncated = self._step_count >= spec.MAX_EPISODE_STEPS

        reward = spec.placeholder_reward(linear, collided)

        return obs, reward, terminated, truncated, {'min_range': min_range}

    def close(self):
        self._cmd_pub.publish(Twist())
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
