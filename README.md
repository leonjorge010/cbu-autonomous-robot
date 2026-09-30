# C.A.$.H Robot Workspace

ROS 2 Jazzy workspace for the C.A.$.H autonomous hallway escort robot.
Runs on Ubuntu 24.04 in WSL2 with Gazebo Harmonic.

## Start working (from computer power-on)

1. **Open Ubuntu**
   - Windows Terminal → click the ⌄ arrow → **Ubuntu 24.04**
   - Or in PowerShell: `wsl -d Ubuntu-24.04`

2. **Go to the workspace and get the latest code**
   ```bash
   cd ~/ros2_ws
   git pull
   ```

3. **Install any new dependencies** (only needed if packages were added)
   ```bash
   rosdep install --from-paths src -y --ignore-src
   ```

4. **Build and source**
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

5. **Open the editor** (optional)
   ```bash
   code .
   ```

6. **Extra terminals**: press `Alt+Shift+D` for a split pane. New terminals source the workspace
   automatically (set up in `~/.bashrc`).

## Run the hallway simulation

```bash
ros2 launch cash_sim sim.launch.py
```

Opens Gazebo (hallway world + robot) and RViz (robot model + LiDAR scan).
**Lighter mode (recommended on the laptop):** `ros2 launch cash_sim sim.launch.py gui:=false` runs Gazebo headless and shows everything in RViz.

Options: `gui:=false`, `rviz:=false`, and spawn pose `x:=… y:=… yaw:=…`.

**Drive it** (in a second terminal):
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Keep that terminal focused and use `i` / `j` / `l` / `,` / `k` to drive.

**The world**: main corridor along +x (2.4 m wide, 30 m long) with a branch going +y at x = 14 m,
classroom doors on the walls, and a bench, water fountain, and trash can as obstacles.
The robot spawns at the origin facing +x.

**ROS topics**

| Topic | Type | Direction |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | ROS → Gazebo |
| `/scan` | `sensor_msgs/LaserScan` (360°, 12 m, 10 Hz) | Gazebo → ROS |
| `/odom` | `nav_msgs/Odometry` | Gazebo → ROS |
| `/tf`, `/joint_states`, `/clock` | TF: `odom → base_footprint → base_link → …` | Gazebo → ROS |

**Files** (`src/cash_sim/`)
- `urdf/cash_bot.urdf.xacro`: robot model. Dimensions at the top are placeholders; update them to the real chassis.
- `worlds/hallway.sdf`: the hallway world.
- `config/bridge.yaml`: Gazebo ↔ ROS topic mapping.
- `launch/sim.launch.py`: starts everything.
- `rviz/cash_sim.rviz`: RViz layout.

## Map the hallway with SLAM Toolbox

One-time setup: `sudo apt update && sudo apt install -y ros-jazzy-slam-toolbox`, then rebuild
(`colcon build --symlink-install && source install/setup.bash`).

1. Start the sim (terminal 1): `ros2 launch cash_sim sim.launch.py gui:=false`
2. Start SLAM (terminal 2): `ros2 launch cash_sim slam.launch.py`
3. Drive around (terminal 3): `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

The growing occupancy grid shows up in RViz (`Map` display, topic `/map`). Drive the full
length of the corridor and the +y branch, and loop back past somewhere you've already been
so slam_toolbox can close the loop and correct drift.

**Save the map** (terminal 4, once you're happy with the coverage):
```bash
ros2 service call /slam_toolbox/save_map slam_toolbox_msgs/srv/SaveMap "{name: {data: '/home/jorge_leon/ros2_ws/maps/hallway'}}"
```
Writes `maps/hallway.pgm` + `maps/hallway.yaml` (used later by Nav2). Map files aren't
committed to git (regenerate by re-mapping); re-run with a different name to keep multiple
attempts.

To keep mapping *state* itself (not just the raster) so you can resume later instead of
re-driving from scratch:
```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox_msgs/srv/SerializePoseGraph "{filename: '/home/jorge_leon/ros2_ws/maps/hallway'}"
```

## Save and push work

```bash
git status
git add .
git commit -m "Describe what changed"
git push
```

## Quick checks

| Check | Command | Expected |
|---|---|---|
| GPU rendering | `glxinfo -B \| grep -i renderer` | `D3D12 (NVIDIA GeForce RTX 4060 ...)` |
| Gazebo | `gz sim shapes.sdf` | Window with shapes opens |
| ROS comms | `ros2 run demo_nodes_cpp talker` + `listener` in a 2nd pane | Listener prints `I heard: ...` |
| GitHub login | `gh auth status` | `Logged in to github.com account leonjorge010` |
| Sim topics | `ros2 topic hz /scan` (sim running) | ~10 Hz |

## Troubleshooting

- **Renderer says `llvmpipe` (slow, CPU rendering)**: make sure `~/.bashrc` contains
  `export GALLIUM_DRIVER=d3d12` and `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`, then `source ~/.bashrc`.
- **`git push` / `gh` asks to log in**: the access token may have expired. Generate a new classic token
  (scopes: repo, workflow, read:org, admin:public_key) and run `gh auth login` → Paste an authentication token.
- **Package not found after building**: run `source install/setup.bash` in that terminal.
- **Robot doesn't move with teleop**: the teleop terminal must be focused, and the sim must be playing (not paused).
- **Slow builds**: make sure you're in `~/ros2_ws`, not under `/mnt/c/...`.
