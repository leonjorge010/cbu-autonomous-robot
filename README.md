# C.A.$.H Robot Workspace

ROS 2 Jazzy workspace (Ubuntu 24.04 / Gazebo Harmonic).

## Build
    cd ~/ros2_ws
    rosdep install --from-paths src -y --ignore-src
    colcon build
    source install/setup.bash
