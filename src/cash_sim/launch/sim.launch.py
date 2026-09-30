"""Launch the C.A.$.H hallway simulation: Gazebo + robot + ROS bridge + RViz."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('cash_sim')
    world = os.path.join(pkg, 'worlds', 'hallway.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'cash_bot.urdf.xacro')
    bridge_config = os.path.join(pkg, 'config', 'bridge.yaml')
    rviz_config = os.path.join(pkg, 'rviz', 'cash_sim.rviz')

    use_rviz = LaunchConfiguration('rviz')
    use_gui = LaunchConfiguration('gui')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    gz_launch = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')

    # gui:=true -> Gazebo server + window. gui:=false -> server only (headless, much lighter).
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={'gz_args': f'-r {world}', 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(use_gui),
    )
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={'gz_args': f'-r -s {world}', 'on_exit_shutdown': 'true'}.items(),
        condition=UnlessCondition(use_gui),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'cash_bot',
                   '-x', x, '-y', y, '-z', '0.02', '-Y', yaw],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Open the Gazebo window (false = headless)'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('x', default_value='0.0', description='Spawn x (m)'),
        DeclareLaunchArgument('y', default_value='0.0', description='Spawn y (m)'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Spawn yaw (rad)'),
        gazebo_gui,
        gazebo_headless,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz,
    ])
