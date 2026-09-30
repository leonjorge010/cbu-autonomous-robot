"""Launch the hallway sim headless plus the service bridge HallwayEnv needs for reset()."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

WORLD_NAME = 'hallway'


def generate_launch_description():
    sim_launch = os.path.join(
        get_package_share_directory('cash_sim'), 'launch', 'sim.launch.py')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={'gui': 'false', 'rviz': 'false'}.items(),
    )

    # Bridges the Gazebo world-control and set-pose services so HallwayEnv.reset()
    # can clear physics state and teleport the robot back to its spawn pose.
    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'/world/{WORLD_NAME}/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        sim,
        service_bridge,
    ])
