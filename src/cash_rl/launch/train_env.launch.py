"""Launch a sim world headless plus the service bridge HallwayEnv needs for reset().

ros2 launch cash_rl train_env.launch.py                     # hallway
ros2 launch cash_rl train_env.launch.py world:=cbu_floor2   # HallwayEnv: CASH_MAP_YAML to match
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROBOT_NAME = 'cash_bot'


def generate_launch_description():
    world = LaunchConfiguration('world')
    sim_launch = os.path.join(
        get_package_share_directory('cash_sim'), 'launch', 'sim.launch.py')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'gui': 'false', 'rviz': 'false', 'world': world,
                          'x': LaunchConfiguration('x'), 'y': LaunchConfiguration('y'),
                          'yaw': LaunchConfiguration('yaw')}.items(),
    )

    # Bridges the Gazebo world-control and set-pose services so HallwayEnv.reset()
    # can clear physics state and teleport the robot to its spawn pose, plus the
    # robot's ground-truth pose (from the URDF's PosePublisher) for goal-relative
    # observations and reward.
    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            ['/world/', world, '/control@ros_gz_interfaces/srv/ControlWorld'],
            ['/world/', world, '/set_pose@ros_gz_interfaces/srv/SetEntityPose'],
            f'/model/{ROBOT_NAME}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='hallway',
                              description='cash_sim world; HallwayEnv must use the matching map '
                                          '(CASH_MAP_YAML=maps/<world>.yaml)'),
        # HallwayEnv teleports the robot on reset; this is only where it first appears
        # (default: the world's spawn, as in sim.launch.py).
        DeclareLaunchArgument('x', default_value=''),
        DeclareLaunchArgument('y', default_value=''),
        DeclareLaunchArgument('yaw', default_value=''),
        sim,
        service_bridge,
    ])
