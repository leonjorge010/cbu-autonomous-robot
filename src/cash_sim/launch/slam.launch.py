"""Launch SLAM Toolbox (online async) against the running C.A.$.H sim.

Run this in a separate terminal after `sim.launch.py` is already up. It
subscribes to /scan and odom->base_footprint tf, and publishes map->odom
plus the /map topic.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('cash_sim')
    slam_config = os.path.join(pkg, 'config', 'slam_toolbox.yaml')

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[slam_config, {'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([slam_toolbox])
