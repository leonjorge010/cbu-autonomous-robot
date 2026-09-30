"""Gymnasium environment wrapping the C.A.$.H hallway sim over ROS 2 topics/services.

Requires `sim.launch.py` (cash_sim) plus this package's service bridge for
/world/hallway/control already running — see launch/train_env.launch.py.
"""
import time

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity, WorldControl, WorldReset
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from sensor_msgs.msg import LaserScan

WORLD_NAME = 'hallway'
ROBOT_NAME = 'cash_bot'
# Must match sim.launch.py's x/y/yaw defaults (0.0) and its hardcoded spawn z (0.02).
SPAWN_POSE = Pose(position=Point(x=0.0, y=0.0, z=0.02), orientation=Quaternion(w=1.0))
N_SCAN_BEAMS = 360
N_OBS_BEAMS = 36
MAX_RANGE = 12.0
COLLISION_RANGE = 0.25
MAX_LINEAR_VEL = 0.5
MAX_ANGULAR_VEL = 1.0
CONTROL_PERIOD = 0.1
MAX_EPISODE_STEPS = 500
SCAN_WAIT_TIMEOUT = 5.0


class HallwayEnv(gym.Env):
    """Scan-in / cmd_vel-out environment for a single cash_bot in hallway.sdf.

    Observation: N_OBS_BEAMS downsampled LiDAR ranges (meters, NaN/inf clipped to
    MAX_RANGE), normalized to [0, 1].
    Action: [linear_x, angular_z] cmd_vel, each in [-1, 1] and scaled to the
    MAX_LINEAR_VEL / MAX_ANGULAR_VEL limits.
    Reward is a placeholder (forward progress minus a collision penalty) —
    replace once escort waypoints/goals exist.
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

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS_BEAMS,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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
        ranges = np.array(self._latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0)
        return np.clip(ranges, 0.0, MAX_RANGE)

    def _observation(self):
        ranges = self._sanitized_ranges()
        stride = N_SCAN_BEAMS // N_OBS_BEAMS
        downsampled = ranges[::stride][:N_OBS_BEAMS]
        return (downsampled / MAX_RANGE).astype(np.float32)

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
        action = np.clip(action, -1.0, 1.0)
        twist = Twist()
        twist.linear.x = float(action[0]) * MAX_LINEAR_VEL
        twist.angular.z = float(action[1]) * MAX_ANGULAR_VEL
        self._cmd_pub.publish(twist)

        self._spin_for(CONTROL_PERIOD)
        self._step_count += 1

        obs = self._observation()
        min_range = float(np.min(self._sanitized_ranges()))
        collided = min_range < COLLISION_RANGE
        terminated = collided
        truncated = self._step_count >= MAX_EPISODE_STEPS

        reward = twist.linear.x * CONTROL_PERIOD
        if collided:
            reward -= 10.0

        return obs, reward, terminated, truncated, {'min_range': min_range}

    def close(self):
        self._cmd_pub.publish(Twist())
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
