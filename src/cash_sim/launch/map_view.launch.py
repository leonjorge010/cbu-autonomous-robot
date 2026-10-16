"""Show a saved map, its keep-out zones and named goals in RViz, aligned with the sim.

Normally started by sim.launch.py's `map:=<name>` argument:

    ros2 launch cash_sim sim.launch.py world:=cbu_floor2 map:=cbu_floor2 gui:=false

Publishes /map and /keepout_mask (nav2_map_server, MAP.yaml and MAP_keepout.yaml from
the workspace's maps/), /goal_markers (MAP_goals.yaml), and a static map -> odom
transform equal to the spawn pose: odom starts where the robot spawned, and the map
frame is the world frame. Wheel odometry drift is not corrected (no localization), and
don't run it alongside slam.launch.py, which publishes its own map -> odom.
"""
import os

from cash_sim.maps import maps_dir
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context):
    name = LaunchConfiguration('map').perform(context)
    map_yaml = os.path.join(maps_dir(), f'{name}.yaml')
    keepout_yaml = os.path.join(maps_dir(), f'{name}_keepout.yaml')
    goals_yaml = os.path.join(maps_dir(), f'{name}_goals.yaml')
    if not os.path.isfile(map_yaml):
        raise FileNotFoundError(map_yaml)
    sim_time = {'use_sim_time': True}
    servers = ['map_server']
    actions = [Node(package='nav2_map_server', executable='map_server', name='map_server',
                    parameters=[{'yaml_filename': map_yaml}, sim_time], output='screen')]
    if os.path.isfile(keepout_yaml):
        servers.append('keepout_map_server')
        actions.append(Node(
            package='nav2_map_server', executable='map_server', name='keepout_map_server',
            parameters=[{'yaml_filename': keepout_yaml, 'topic_name': '/keepout_mask'},
                        sim_time], output='screen'))
    actions.append(Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='map_view_lifecycle_manager',
        parameters=[{'autostart': True, 'node_names': servers}, sim_time], output='screen'))
    if os.path.isfile(goals_yaml):
        actions.append(Node(package='cash_sim', executable='goal_markers',
                            parameters=[{'goals_yaml': goals_yaml}], output='screen'))
    x, y, yaw = (LaunchConfiguration(k).perform(context) for k in ('x', 'y', 'yaw'))
    actions.append(Node(
        package='tf2_ros', executable='static_transform_publisher', name='map_to_odom',
        arguments=['--x', x, '--y', y, '--yaw', yaw, '--frame-id', 'map',
                   '--child-frame-id', 'odom'],
        parameters=[sim_time], output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', description='map name in <workspace>/maps, e.g. cbu_floor2'),
        DeclareLaunchArgument('x', default_value='0.0', description='robot spawn x (m)'),
        DeclareLaunchArgument('y', default_value='0.0', description='robot spawn y (m)'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='robot spawn yaw (rad)'),
        OpaqueFunction(function=launch_setup),
    ])
