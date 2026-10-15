# C.A.$.H Robot Workspace — Claude Code context

## Project
CBU Autonomous Student Helper (C.A.$.H): an autonomous robot that escorts students to classrooms in the CBU Engineering building using SLAM navigation. Capstone team: Lancer Automation Systems. The real robot runs on a Jetson Orin Nano (needs JetPack 7.2+ for Ubuntu 24.04 / native Jazzy; JetPack 6 is 22.04 / Humble) with a 2D LiDAR only (no depth or RGB camera). This workspace is the simulation and software side; a DRL hallway-navigation policy will be trained here in simulation (CSCI 4220 project).

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
- `urdf/cash_bot.urdf.xacro`: diff-drive robot (0.50×0.40 m base, wheels at center, front/rear casters, 360° gpu_lidar 12 m @10 Hz). Dimensions are placeholders for the real chassis. A sim-only PosePublisher plugin publishes ground-truth model pose on gz `/model/cash_bot/pose` (bridged by `cash_rl`'s launch, not `bridge.yaml`).
- `worlds/hallway.sdf`: 30 m × 2.4 m corridor with a +y branch at x=14, doors, bench, fountain, trash can. Shadows off for performance.
- `config/bridge.yaml`: /clock, /cmd_vel, /odom, /tf, /joint_states, /scan
- `config/slam_toolbox.yaml`: online async SLAM params (odom_frame odom, base_frame base_footprint, scan_topic /scan)
- `launch/sim.launch.py`: args `gui`, `rviz`, `x`, `y`, `yaw`
- `launch/slam.launch.py`: starts `async_slam_toolbox_node`; run in a second terminal after `sim.launch.py` is up
- `rviz/cash_sim.rviz`: fixed frame `odom`; includes a `/map` display for use while SLAM is running
- TF: odom → base_footprint → base_link → wheels/casters/lidar_link; SLAM Toolbox adds map → odom on top
- `maps/`: SLAM Toolbox output (`.pgm`/`.yaml` via `/slam_toolbox/save_map`, `.posegraph`/`.data` via `/slam_toolbox/serialize_map`). Final maps are committed; name throwaway attempts `maps/scratch_*` (gitignored).

## Package: src/cash_rl (ament_python)
- `cash_rl/spec.py`: single source of truth (no ROS imports) for the goal-reaching task, mirroring the URDF (keep in sync). Both envs import it, so a policy trained in one runs in the other unchanged. Observation (41): 36 of the 360 beams / 12 m, goal distance / 30 m + cos/sin of goal bearing in the robot frame, previous action. Action: [-1,1]² → 0.5 m/s, 1.0 rad/s. Collision: any return inside the 0.50×0.49 m footprint (width = wheels' outer faces) inflated by 0.05 m. Reward per step: 10 × decrease in *geodesic* goal distance − 0.01, +10 on reaching (within 0.3 m), −20 on collision; geodesic so the T-junction's inner corner isn't a local optimum. Step limit scales with start distance (2× full-speed time + 10 s). `DOOR_GOALS`: corridor-centerline points in front of each door in `hallway.sdf`, the fixed eval goal set (and seed for milestone 4).
- `cash_rl/occupancy_map.py`: loads the SLAM map (`$CASH_MAP_YAML`, else `maps/hallway.yaml` found by walking up from the source file — works with symlink install) and computes geodesic distance fields (Dijkstra on cells ≥0.25 m from obstacles, ~12 ms per goal).
- `cash_rl/raycast_env.py`: `HallwayRaycastEnv`, ROS-free 2D training env — unicycle model with the DiffDrive accel limits, raycasts the observed beams against the map (~2.3k steps/s single core). `reset(options={'spawn': (x, y, yaw), 'goal': (x, y)})`; omitted ones are sampled (random collision-free spawn, goal 1–15 m geodesic away). `info['is_success']` for SB3's EvalCallback; `render_mode='rgb_array'`. Checked against live Gazebo `/scan`: median beam error ~6 cm, which is the SLAM map's ±1 cell wall placement, not the raycaster.
- `cash_rl/hallway_env.py`: `HallwayEnv`, the same task in Gazebo over rclpy (`/scan`, `/cmd_vel`, ground-truth `/model/cash_bot/pose` — odom drifts ~15 m per mapping pass, so it isn't used). reset() calls ControlWorld (model_only reset) then SetEntityPose to teleport to `options['spawn']` (default `spec.SPAWN_POSE`); goal defaults to a random `DOOR_GOALS` entry. Steps on wall-clock time (0.1 s), not sim time.
- `cash_rl/view_env.py` (console script `view_env`): plays `HallwayRaycastEnv` episodes in a matplotlib window (or `--save x.gif`) — map to scale, footprint, beams, goal, trail, step/return/outcome. `--policy downhill|random`, `--goal random|all|door_XXX`. GUI: ask me to run it, don't launch it yourself.
- `cash_rl/demo.py` (console script `demo`): interactive 2D demo — left-click places the robot (drag to aim; a plain click faces +x), right-click sets the destination (or 1–7 for a door), Esc clears; chains legs from where the robot stopped. Defaults to the newest `runs/*/best_model.zip`. Drawing shared with `view_env` via `view_env.Scene`. GUI: ask me to run it.
- `cash_rl/scripted.py`: `downhill`, a hand-written reference controller that cheats (reads the map's geodesic field); used by tests and `view_env` as a baseline.
- `cash_rl/train.py` (console script `train`): SB3 training on `HallwayRaycastEnv` (`--algo ppo|sac|td3`, default PPO 3M steps (~38 min on this laptop), 12 `SubprocVecEnv`s, `--goal-distance-range MIN MAX` for training goals; eval always uses the default range so runs stay comparable). Writes `runs/<name>/` (gitignored): `config.json`, TensorBoard logs, `eval.csv`, `checkpoints/`, `best_model.zip`, `final_model.zip`. `GoalEvalCallback` evaluates deterministically every 100k steps on all `DOOR_GOALS` from spawn + 20 fixed-seed random pairs, logging success/collision/timeout rate and return (`eval/*`) — use those curves in the report, not training reward. `view_env --policy runs/<name>/best_model.zip` watches a trained model.
- `cash_rl/evaluate.py` (console script `evaluate`): runs a trained model on every `DOOR_GOALS` entry from spawn and prints per-door outcome/steps/return; `--env 2d` (default) or `--env gazebo` (needs `train_env.launch.py`). Run Gazebo evals with training stopped — `HallwayEnv` steps on wall-clock time, and training load drops Gazebo's real-time factor to 0.6–2×.
- `cash_rl/random_agent.py` (console script `random_agent`): drives `HallwayEnv` with random actions for a few episodes — smoke test for reset/step/service-bridge before wiring up real training.
- `launch/train_env.launch.py`: includes `cash_sim`'s `sim.launch.py` headless plus a `ros_gz_bridge` for the `/world/hallway/control` and `/world/hallway/set_pose` services and the `/model/cash_bot/pose` ground-truth topic (none are in `cash_sim/config/bridge.yaml`).
- Runtime deps `gymnasium`/`numpy` are pip, not rosdep-resolvable here — installed via `pip install --user --break-system-packages gymnasium` (numpy stayed at the system 1.26.4).
- Episode reset calls `ControlWorld` (world reset, clears sim time/physics state) *and* `SetEntityPose` targeting the `cash_bot` entity by name — `ControlWorld`'s `model_only` reset alone does not restore pose for entities spawned dynamically after world load (confirmed by smoke test: episodes collapsed to 1-4 steps because the robot never left the wall it crashed into), so the explicit teleport is required, not just a nice-to-have.

## Conventions
- Installed via pip (user site): CPU-only `torch` (from the PyTorch CPU index), `stable-baselines3`, `tensorboard`. This also put `setuptools` 78 and `protobuf` 7 in the user site, shadowing the apt versions; colcon builds and ROS imports were checked and still work.
- Install Python packages with `pip install --user --break-system-packages`; when installing `torch` or `stable-baselines3`, always pin `"numpy<2"` alongside them so they don't upgrade the system numpy (1.26.4) that ROS 2 Jazzy's Python packages expect
- Jazzy + Gazebo Harmonic only: `ros_gz` and `gz-sim-*` system plugins, never Gazebo Classic / `gazebo_ros`
- Absolute topic names; `use_sim_time: true` on all sim nodes
- Never commit `build/`, `install/`, `log/`, rosbags, or trained model weights
- Small commits with clear messages
- Add a dated section to `WORKLOG.md` (newest first) at the end of each work session

## Known harmless output
- `gz_frame_id ... not defined in SDF` warning at spawn
- `Segmentation fault` / Ruby `Errno::ESRCH` from Gazebo when stopping with Ctrl+C
- RViz "Message Filter dropping message ... earlier than all the data in the transform cache" right after sim start

## Next milestones
1. ~~SLAM Toolbox: map the hallway~~ — done: `maps/hallway.{pgm,yaml,posegraph,data}` (0.05 m/px, map frame aligned with Gazebo world frame since SLAM started at spawn). If a Ctrl+C leaves an orphan `gz sim` server running, a relaunch silently reuses its stale robot state — check `pgrep -af "gz sim"` before mapping
2. Nav2: load the saved map with `nav2_map_server`, localize against it (AMCL or slam_toolbox in localization mode), and use Nav2's planner/controller to drive to goal poses set in RViz
3. DRL navigation policy: train a hallway-navigation policy headless in this sim (CSCI 4220 project) — LiDAR scan + relative goal as observation, `/cmd_vel` as the action space; an alternative/complement to the classic Nav2 stack for the escort behavior. Done: `cash_rl` goal-reaching task (shared spec, Gazebo `HallwayEnv`, fast 2D `HallwayRaycastEnv`, geodesic-progress reward). Training script done; first run `runs/ppo_baseline` (PPO, 3M steps): eval success 0% → ~85% by 1M, best 93% (7/7 doors, 0% collisions) at 2.8M; that model reached 7/7 doors in Gazebo with no collisions, returns within 0.2 of 2D (Gazebo episodes take 1–6% fewer steps). `runs/ppo_junction` (`--goal-distance-range 1 25 --junction-prob 0.3`, 1.5M steps) is the current best: 96% success / 3% collisions on 200 held-out goals vs baseline 93% / 6% (one seed each). Both models are published as GitHub release `models-2026-10-14`. Open: multi-seed repeats, reversing (~51% of steps from random headings), fountain clips, then random static obstacles → pedestrians → unseen layout, Gazebo eval from varied spawns, numpy export of the actor for the Jetson
4. Escort behavior: turn the doors already in `hallway.sdf` into named waypoints so the robot can be given "escort to classroom X" goals
5. Port whatever works in sim to the real Jetson + 2D LiDAR hardware — the sim was built with no depth/RGB camera specifically to keep sim and real perception matched
