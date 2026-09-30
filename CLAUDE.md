# C.A.$.H Robot Workspace — Claude Code context

## Project
CBU Autonomous Student Helper (C.A.$.H): an autonomous robot that escorts students to classrooms in the CBU Engineering building using SLAM navigation. Capstone team: Lancer Automation Systems. The real robot runs on a Jetson with a 2D LiDAR only (no depth or RGB camera). This workspace is the simulation and software side; a DRL hallway-navigation policy will be trained here in simulation (CSCI 4220 project).

## Environment
- Windows 11 → WSL2 Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (gz sim 8) via ros_gz
- GPU: RTX 4060 Laptop via WSLg D3D12. `~/.bashrc` sets `GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. If `glxinfo -B` shows `llvmpipe`, rendering fell back to CPU.
- Workspace: `~/ros2_ws` (Linux filesystem, never `/mnt/c`)
- Repo: github.com/leonjorge010/cbu-autonomous-robot (branch `main`)

## Build
- `colcon build --symlink-install` then `source install/setup.bash`
- Python/launch/URDF/world edits take effect without rebuilding (symlink install); new files or setup.py changes need a rebuild
- If colcon errors about missing files in `install/` after changing build modes: `rm -rf build install log` and rebuild

## Run
- Preferred: `ros2 launch cash_sim sim.launch.py gui:=false` (headless Gazebo + RViz). Gazebo GUI + RViz together makes the laptop freeze.
- Gazebo window only: `gui:=true rviz:=false`
- Drive: `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
- Do not launch the simulation or GUI apps yourself; ask me to run them in another terminal. Use `ros2 topic list/echo/hz`, `ros2 node list`, and `ros2 param` to inspect a running sim.

## Package: src/cash_sim (ament_python)
- `urdf/cash_bot.urdf.xacro`: diff-drive robot (0.50×0.40 m base, wheels at center, front/rear casters, 360° gpu_lidar 12 m @10 Hz). Dimensions are placeholders for the real chassis.
- `worlds/hallway.sdf`: 30 m × 2.4 m corridor with a +y branch at x=14, doors, bench, fountain, trash can. Shadows off for performance.
- `config/bridge.yaml`: /clock, /cmd_vel, /odom, /tf, /joint_states, /scan
- `config/slam_toolbox.yaml`: online async SLAM params (odom_frame odom, base_frame base_footprint, scan_topic /scan)
- `launch/sim.launch.py`: args `gui`, `rviz`, `x`, `y`, `yaw`
- `launch/slam.launch.py`: starts `async_slam_toolbox_node`; run in a second terminal after `sim.launch.py` is up
- `rviz/cash_sim.rviz`: fixed frame `odom`; includes a `/map` display for use while SLAM is running
- TF: odom → base_footprint → base_link → wheels/casters/lidar_link; SLAM Toolbox adds map → odom on top
- `maps/`: SLAM Toolbox output (`.pgm`/`.yaml` via `/slam_toolbox/save_map`, `.posegraph`/`.data` via `/slam_toolbox/serialize_map`). Final maps are committed; name throwaway attempts `maps/scratch_*` (gitignored).

## Conventions
- Jazzy + Gazebo Harmonic only: `ros_gz` and `gz-sim-*` system plugins, never Gazebo Classic / `gazebo_ros`
- Absolute topic names; `use_sim_time: true` on all sim nodes
- Never commit `build/`, `install/`, `log/`, rosbags, or trained model weights
- Small commits with clear messages

## Known harmless output
- `gz_frame_id ... not defined in SDF` warning at spawn
- `Segmentation fault` / Ruby `Errno::ESRCH` from Gazebo when stopping with Ctrl+C
- RViz "Message Filter dropping message ... earlier than all the data in the transform cache" right after sim start

## Next milestones
1. ~~SLAM Toolbox: map the hallway~~ — wired up (`slam.launch.py`); still need to actually drive a full mapping pass and save `maps/hallway.*`
2. Nav2: load the saved map with `nav2_map_server`, localize against it (AMCL or slam_toolbox in localization mode), and use Nav2's planner/controller to drive to goal poses set in RViz
3. DRL navigation policy: train a hallway-navigation policy headless in this sim (CSCI 4220 project) — LiDAR scan as observation, `/cmd_vel` as the action space; an alternative/complement to the classic Nav2 stack for the escort behavior
4. Escort behavior: turn the doors already in `hallway.sdf` into named waypoints so the robot can be given "escort to classroom X" goals
5. Port whatever works in sim to the real Jetson + 2D LiDAR hardware — the sim was built with no depth/RGB camera specifically to keep sim and real perception matched
