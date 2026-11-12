# CBU Autonomous Robot

Simulation and navigation software for the **CBU Autonomous Robot**, a robot that
escorts students to classrooms in the CBU Engineering building. The real robot is a Jetson Orin Nano
with a single 2D LiDAR, so the simulated robot has exactly that sensor and nothing else.

Stack: Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic · Stable-Baselines3 (PyTorch, CPU)

## How the model is developed

```
 1. Simulate      2. Map           3. Train in 2D        4. Check in Gazebo    5. Robot
 Gazebo hallway → SLAM Toolbox  → fast raycast sim    → same policy, full   → Jetson +
 + LiDAR robot    occupancy map   built on that map     physics sim           real LiDAR
                                  (~2,300 steps/s)      (~10 steps/s)
```

1. **Simulate.** `cash_sim` has two Gazebo worlds and a differential-drive robot with a 360°
   LiDAR modeled on the real one (LDROBOT STL-27L: 25 m range, 10 Hz). `hallway` is a 30 m
   corridor with a side branch, classroom doors, a bench, a water fountain and a trash can.
   `cbu_floor2` is the real TEGR 2nd floor (~140 m of corridor).
2. **Map.** We drove the robot through the hallway with SLAM Toolbox and saved the map in `maps/`.
   This is the same thing the real robot will do in the real building.
3. **Train.** Gazebo is far too slow for reinforcement learning, so `cash_rl` has a lightweight 2D
   simulator that casts the LiDAR beams against the saved map. PPO learns to drive from a start
   pose to a goal point.
4. **Validate.** The trained policy runs unchanged in Gazebo, because both simulators share one task
   definition (`src/cash_rl/cash_rl/spec.py`).
5. **Deploy** (not started). The policy gets exported for the Jetson and runs on the real LiDAR.

**The learning task**

| | |
|---|---|
| Observation (113 numbers) | 36 LiDAR distances (every 10°) from each of the last 3 scans, distance and direction to the goal (or to a subgoal ahead on the route: 1.5 m on the current TEGR models, 3 m on the earlier ones), last action |
| Action | forward speed (up to 0.5 m/s; reverse only 0.15 m/s) and turn rate (up to 1 rad/s) |
| Reward | + progress toward the goal measured *around* walls (with extra cost near walls), − small time penalty, +10 for reaching the goal, −20 for a collision (−40 for hitting a pedestrian, plus penalties for crowding one) |
| Success | within 0.3 m of the goal |
| Evaluation | every 100k steps: all 7 doors from the spawn point + 20 fixed random start/goal pairs |

**Results so far.** The full comparison, with every training parameter and learning curves, is in
[docs/MODELS.md](docs/MODELS.md).

Hallway, no pedestrians (200 unseen goals):

| Model | Sensor model | Goals reached | Collisions | Time spent reversing |
|---|---|---|---|---|
| `ppo_junction` | ideal LiDAR, fast reverse | 96% | 3% | 51% |
| `ppo_d800_prox` (released, mean of 3 seeds) | real STL-27L LiDAR, slow reverse, keeps clear of walls | 93% ± 3% | 4% ± 2% | 10% |

With the LiDAR safety layer (`cash_rl/safety.py`, `evaluate --safety`) in front of the policy,
collisions drop to 0.7% on average, with no loss in success. All models reach all 7 doors.

Real TEGR 2nd floor (`cbu_floor2`, 200 unseen goals). Older rows are the encounter test in
[docs/MODELS.md](docs/MODELS.md): one pedestrian walks down the robot's route toward it.

| Model | No pedestrians | Encounter test |
|---|---|---|
| Hallway model, never trained on TEGR (subgoal 3 m ahead, 3 seeds) | 72.5% (57% without the subgoal) | — |
| `tegr_ppo` (PPO, 3 seeds) | 84% | 42% |
| `tegr_dagger_sg3` + safety layer (copies a scripted avoider, 3 seeds) | **95%** | **73%** |
| `tegr_ppo_lane` (3 m subgoal, 1 seed) | 94% | 47.5% center lane, 51% with the passing rule |
| `tegr_ppo_la15` (1.5 m subgoal, lanes held 8–15 s, 3 seeds) | **97%** | 69% mean with the passing rule |

The lane retrain is what lets the passing rule help. The current learned driver is
`tegr_dagger_dh_s0`: that `la15` policy imitating the scripted lane follower while the rule
picks the lane. The weights are not in git (`runs/` is gitignored). On one identical test
(12 Nov, subgoal 1.5 m, passing side held for the whole pass):

| | Empty corridor | One person, head-on | One person, mixed | Four people, head-on | Four people, mixed |
|---|---|---|---|---|---|
| `tegr_ppo_la15_s0` | 85.5% | 82% | 76.5% | 75.5% | 65% |
| `tegr_dagger_dh_s0` | **96.5%** | **83%** | **81%** | 75.5% | 65% |
| Scripted lane follower | — | **86.5%** | 80% | 77% | 69% |

The empty-corridor column is a center subgoal. `la15` reaches 97% when it is allowed the
random lanes it trained on, and 85.5% when the lane stays in the center. With the rule turned
off, both learned models reach the goal about 41% of the time and hit something on most of the
rest. The rule does the passing. The learned model has caught it on one person and on a crowd,
and it is ahead on an empty corridor. Most of the scripted follower's remaining crashes are a
lane the rule called clear. Collision and time numbers are in `WORKLOG.md` (12 Nov).

## Setup (once per computer)

On Windows, first install WSL2 with Ubuntu 24.04 (`wsl --install -d Ubuntu-24.04` in PowerShell), then
do everything below inside Ubuntu.

1. Install [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
   (`ros-jazzy-desktop`) and `ros-jazzy-ros-gz`, which also installs Gazebo Harmonic.
2. Clone into `~/ros2_ws` (the repo *is* the workspace). On WSL, keep it in the Linux home folder,
   not under `/mnt/c`, because builds there are very slow.
   ```bash
   git clone https://github.com/leonjorge010/cbu-autonomous-robot.git ~/ros2_ws
   cd ~/ros2_ws
   ```
3. Install dependencies and build:
   ```bash
   source /opt/ros/jazzy/setup.bash
   sudo rosdep init 2>/dev/null; rosdep update
   rosdep install --from-paths src -y --ignore-src
   pip install --user --break-system-packages --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
   colcon build --symlink-install
   ```
4. Source the workspace automatically in new terminals:
   ```bash
   echo 'source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
   ```
5. WSL with an NVIDIA GPU only: add these lines to `~/.bashrc` so Gazebo renders on the GPU. Check
   with `glxinfo -B`: the renderer should be D3D12, not `llvmpipe`.
   ```bash
   export GALLIUM_DRIVER=d3d12
   export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
   ```

Day to day: `cd ~/ros2_ws && git pull && colcon build --symlink-install`. You only need to rebuild
when files are added or `setup.py` changes.

## Run

**Simulation.** `gui:=false` runs Gazebo without its window and shows everything in RViz. Opening
both windows can freeze a laptop.
```bash
ros2 launch cash_sim sim.launch.py gui:=false                              # hallway
ros2 launch cash_sim sim.launch.py world:=cbu_floor2 gui:=false            # TEGR 2nd floor
ros2 run teleop_twist_keyboard teleop_twist_keyboard     # 2nd terminal; drive with i j l , k
```

**Train a policy.** Takes about 19 min. Output goes to `runs/<name>/`, which is not committed.
```bash
ros2 run cash_rl train --name my_run --timesteps 1500000 --goal-distance-range 1 25 --junction-prob 0.3
tensorboard --logdir ~/ros2_ws/runs    # open http://localhost:6006 and look at the eval/ charts
```

**Or download the trained models** from the
[`models-2026-10-15` release](https://github.com/leonjorge010/cbu-autonomous-robot/releases/tag/models-2026-10-15).
Each run also has its `config.json` (every setting) and `eval.csv` (learning curve).
```bash
cd ~/ros2_ws
for run in ppo_d800_prox ppo_d800_wallcost ppo_d800; do
  mkdir -p runs/$run
  curl -L -o runs/$run/best_model.zip \
    https://github.com/leonjorge010/cbu-autonomous-robot/releases/download/models-2026-10-15/${run}_best_model.zip
done
```
The older `ppo_baseline` and `ppo_junction` models
([`models-2026-10-14`](https://github.com/leonjorge010/cbu-autonomous-robot/releases/tag/models-2026-10-14))
were trained on the ideal sensor with fast reverse. Run them at commit `5cb4ade`.

**Try a trained model** (2D, no Gazebo needed):
```bash
ros2 run cash_rl demo                  # click to place the robot, right-click a destination, keys 1-7 = doors
ros2 run cash_rl view_env --policy runs/my_run/best_model.zip --goal all   # add --save out.gif to record
ros2 run cash_rl evaluate runs/my_run/best_model.zip                      # per-door results
ros2 run cash_rl evaluate runs/a/best_model.zip runs/b/best_model.zip --random 200   # compare models
ros2 run cash_rl evaluate runs/tegr_dagger_dh_s0/final_model.zip \
    --map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1 \
    --subgoal-lookahead 1.5 --decision-layer --wait
```

**Evaluate in Gazebo.** Stop any training first, because Gazebo runs on wall-clock time.
```bash
ros2 launch cash_rl train_env.launch.py                                 # terminal 1
ros2 run cash_rl evaluate runs/my_run/best_model.zip --env gazebo       # terminal 2
```

**Re-map the hallway** (only if the world changes):
```bash
ros2 launch cash_sim sim.launch.py gui:=false     # terminal 1
ros2 launch cash_sim slam.launch.py               # terminal 2, then drive the whole hallway
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '$HOME/ros2_ws/maps/hallway'}}"
```

## Where things are

| Path | What it is |
|---|---|
| `src/cash_sim/worlds/hallway.sdf` | The Gazebo hallway |
| `src/cash_sim/worlds/cbu_floor2.sdf` | The Gazebo TEGR 2nd floor, generated from the floorplan |
| `src/cash_sim/urdf/cash_bot.urdf.xacro` | The robot. Dimensions are placeholders for the real chassis |
| `src/cash_sim/launch/` | `sim.launch.py` (simulation), `slam.launch.py` (mapping) |
| `maps/hallway.*`, `maps/cbu_floor2.*` | Saved maps. Training and eval use these, not the live Gazebo physics |
| `src/cash_rl/cash_rl/spec.py` | **The task**: observation, action, reward, collision. Shared by both simulators |
| `src/cash_rl/cash_rl/raycast_env.py` | Fast 2D training simulator |
| `src/cash_rl/cash_rl/decision.py` | Passing rule: picks the lane around a person. The motion net follows that lane |
| `src/cash_rl/cash_rl/hallway_env.py` | The same task in Gazebo |
| `src/cash_rl/cash_rl/train.py` | Training script. PPO hyperparameters are in `algo_kwargs` |
| `docs/MODELS.md` | Comparison of models through the Oct 31 lane and subgoal runs |
| `WORKLOG.md` | What was done each session, with numbers. Nov 12 has the current comparison |
| `lab/` | Compute-lab job scripts for the TEGR runs. Not needed to simulate or evaluate locally |

ROS topics: `/cmd_vel` (drive), `/scan` (LiDAR, 10 Hz), `/odom`, `/tf`, `/clock`.

## Contributing

- Make small commits with clear messages, and add a dated entry to `WORKLOG.md` after each session.
- Never commit `build/`, `install/`, `log/`, `runs/` (model weights) or rosbags.
- Training changes should be reported with the `eval/` numbers, not the training reward.

## Troubleshooting

- **`cd ~/ros2_ws` fails or VS Code opens the wrong folder**: you're in PowerShell, not Ubuntu. Run
  `wsl -d Ubuntu-24.04` (and `wsl --set-default Ubuntu-24.04` so this sticks).
- **Package not found**: run `source install/setup.bash`, or rebuild if you just pulled new files.
- **Robot spawns in a weird place after a relaunch**: an old Gazebo server survived Ctrl+C. Check with
  `pgrep -af "gz sim"`, `kill -9` any leftovers, and relaunch.
- **Gazebo is very slow**: check `glxinfo -B`. If the renderer is `llvmpipe`, see setup step 5.
- **Harmless messages**: `gz_frame_id ... not defined`, a segfault when Gazebo exits, and RViz
  "Message Filter dropping message" right after startup.
