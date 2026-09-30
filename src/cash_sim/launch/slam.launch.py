"""Launch SLAM Toolbox (online async) against the running C.A.$.H sim.

Run this in a separate terminal after `sim.launch.py` is already up. It
subscribes to /scan and odom->base_footprint tf, and publishes map->odom
plus the /map topic.

async_slam_toolbox_node is a lifecycle node: it starts "unconfigured" and
must be driven through configure -> activate before it does anything. The
event handlers below do that automatically on launch (same pattern as
slam_toolbox's own online_async_launch.py).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    pkg = get_package_share_directory('cash_sim')
    slam_config = os.path.join(pkg, 'config', 'slam_toolbox.yaml')

    slam_toolbox = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',
        parameters=[slam_config, {'use_sim_time': True}],
        output='screen',
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_toolbox),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        ),
    )

    return LaunchDescription([slam_toolbox, configure_event, activate_event])
