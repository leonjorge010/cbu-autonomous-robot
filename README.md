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
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/home/jorge_leon/ros2_ws/maps/hallway'}}"
```
Writes `maps/hallway.pgm` + `maps/hallway.yaml` (used later by Nav2). Commit good maps —
they're checked in. Use a `maps/scratch_*` name for throwaway test runs; those are
gitignored.

To keep mapping *state* itself (not just the raster) so you can resume later instead of
re-driving from scratch:
```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/jorge_leon/ros2_ws/maps/hallway'}"
```

The committed map (`maps/hallway.*`) is aligned with the Gazebo world frame, because SLAM was started
at the spawn pose, so map coordinates equal world coordinates from `hallway.sdf`.

## Train the navigation policy (DRL)

The `cash_rl` package trains a policy that drives from a start pose to a goal point (for example a
classroom door) using only the LiDAR scan and the goal's relative position. Training runs in a fast
2D simulator built on the SLAM map (~2,300 steps/s vs ~10 in Gazebo); the trained policy then runs
unchanged in Gazebo.

One-time setup (CPU-only PyTorch; `numpy<2` keeps ROS 2 Jazzy's numpy working):
```bash
pip install --user --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch "numpy<2"
pip install --user --break-system-packages stable-baselines3 tensorboard gymnasium "numpy<2"
```

**Train** (results go to `runs/<name>/`, which is gitignored; ~1,300 steps/s on the laptop CPU):
```bash
# recommended settings (best so far, ~19 min):
ros2 run cash_rl train --name my_run --timesteps 1500000 --goal-distance-range 1 25 --junction-prob 0.3
ros2 run cash_rl train --help           # all options (algo, seed, eval frequency, ...)
tensorboard --logdir ~/ros2_ws/runs     # then open http://localhost:6006, look at the eval/ charts
```
Every 100k steps the policy is evaluated on all 7 doors from spawn plus 20 fixed random goals;
`runs/<name>/eval.csv` has success, collision and timeout rates for plotting, and
`runs/<name>/config.json` records every setting the run used. `--junction-prob` is the share of
training episodes that start on one side of the junction with the goal on the other.

Where to change things: training-run settings are `train` flags; PPO hyperparameters are in
`algo_kwargs` in `src/cash_rl/cash_rl/train.py`; the task itself (reward weights, goal tolerance,
robot and LiDAR limits) is in `src/cash_rl/cash_rl/spec.py`, shared by the 2D sim and Gazebo.

**Demo** a trained model interactively (2D, no Gazebo needed):
```bash
ros2 run cash_rl demo                   # uses the newest runs/*/best_model.zip
```
Left-click places the robot (drag to aim it), right-click sets the destination, keys 1–7 send it to
a door, Esc clears. After it arrives, right-click again to continue from where it stopped.

**Watch** episodes in a window (or add `--save out.gif`):
```bash
ros2 run cash_rl view_env --policy ~/ros2_ws/runs/my_run/best_model.zip --goal all
ros2 run cash_rl view_env                  # hand-written reference controller, random goals
```

**Evaluate** on the door goals, in 2D or in Gazebo, and compare models on held-out goals:
```bash
ros2 run cash_rl evaluate ~/ros2_ws/runs/my_run/best_model.zip
ros2 run cash_rl evaluate runs/run_a/best_model.zip runs/run_b/best_model.zip --random 200
ros2 launch cash_rl train_env.launch.py                                    # terminal 1, for Gazebo
ros2 run cash_rl evaluate ~/ros2_ws/runs/my_run/best_model.zip --env gazebo # terminal 2
```
Run Gazebo evaluations with training stopped: Gazebo steps on wall-clock time, and training load
slows the simulation. To watch, run `gz sim -g` in another terminal (not together with RViz).

**Results so far** (Oct 14 2026; details in [WORKLOG.md](WORKLOG.md)):

| Run | Settings | Best model | Held-out goals reached (200) | Collisions |
|---|---|---|---|---|
| `ppo_baseline` | 3M steps, goals 1–15 m | 2.8M steps | 93.0% | 6.0% |
| `ppo_junction` | 1.5M steps, goals 1–25 m, `--junction-prob 0.3` | 1.1M steps | 96.0% | 3.0% |

Both best models reach all 7 doors from spawn; `ppo_baseline`'s also did in Gazebo. Known
weaknesses: from random headings both models drive in reverse about half the time instead of turning
around, and the policy can clip the water fountain when it hugs the north wall heading for the branch.

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
- **Robot starts in a weird place / odom is wrong after relaunching**: Ctrl+C sometimes leaves an old
  Gazebo server running, and the relaunch silently reuses it. Check with `pgrep -af "gz sim"`; if more
  than one shows up (or any after stopping the sim), `kill -9 <pid>` the extras and relaunch. Always check
  this before a SLAM mapping pass.
- **Robot doesn't move with teleop**: the teleop terminal must be focused, and the sim must be playing (not paused).
- **Slow builds**: make sure you're in `~/ros2_ws`, not under `/mnt/c/...`.
- **`code .` opens the wrong folder, or `cd ~/ros2_ws` says "path not found"**: you're in Windows
  PowerShell, not WSL Ubuntu — this project only exists inside WSL. Run `wsl -d Ubuntu-24.04` to switch
  (plain `wsl` can land you in Docker Desktop's internal distro instead if that's your current default —
  check with `wsl -l -v`, and fix it with `wsl --set-default Ubuntu-24.04`). Then `cd ~/ros2_ws && code .`
  from that shell. VS Code should show "WSL: Ubuntu-24.04" in the bottom-left corner.
