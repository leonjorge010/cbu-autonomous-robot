"""Publish a map's named goals (MAP_goals.yaml) as RViz markers on /goal_markers.

    ros2 run cash_sim goal_markers --ros-args -p goals_yaml:=/path/to/MAP_goals.yaml

An arrow per goal (pointing at the door when the goal has a yaw, else a sphere),
a text label, and a green arrow at the spawn. Latched (transient local), in the
map frame.
"""
import math

from geometry_msgs.msg import Point, Quaternion
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
import yaml


def _marker(ns, i, kind, x, y, yaw, rgba, scale):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns, m.id, m.type, m.action = ns, i, kind, Marker.ADD
    m.pose.position = Point(x=float(x), y=float(y), z=0.05)
    m.pose.orientation = Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    return m


def markers(goals):
    out = MarkerArray()
    for i, (name, pose) in enumerate(sorted((goals.get('goals') or {}).items())):
        x, y = pose[0], pose[1]
        if len(pose) > 2:
            out.markers.append(_marker('goals', i, Marker.ARROW, x, y, float(pose[2]),
                                       (1.0, 0.55, 0.0, 0.9), (0.6, 0.12, 0.12)))
        else:
            out.markers.append(_marker('goals', i, Marker.SPHERE, x, y, 0.0,
                                       (1.0, 0.55, 0.0, 0.9), (0.3, 0.3, 0.3)))
        label = _marker('labels', i, Marker.TEXT_VIEW_FACING, x, y + 0.45, 0.0,
                        (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.45))
        label.text = name
        out.markers.append(label)
    if goals.get('spawn'):
        x, y, yaw = goals['spawn']
        out.markers.append(_marker('spawn', 0, Marker.ARROW, x, y, float(yaw),
                                   (0.2, 0.9, 0.2, 0.9), (0.8, 0.15, 0.15)))
    return out


def main():
    rclpy.init()
    node = rclpy.create_node('goal_markers')
    path = node.declare_parameter('goals_yaml', '').value
    with open(path) as f:
        goals = yaml.safe_load(f) or {}
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(MarkerArray, '/goal_markers', qos)
    msg = markers(goals)
    publisher.publish(msg)
    node.get_logger().info(f'Published {len(goals.get("goals") or {})} goals from {path}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
