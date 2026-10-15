# Work log

What was done on each work day, newest first. Add a new `## YYYY-MM-DD` section at the top
for each session: what changed, results with numbers, and what's still open.

## 2026-10-15

**Sharing with the team**
- Rewrote the README around the pipeline, first-time setup and how to run. Added
  `requirements.txt` and the missing matplotlib dependency so a fresh clone works.
- Published the 10-14 models as GitHub release `models-2026-10-14`, and gave the repo a
  description and topics. Renamed the project to "CBU Autonomous Robot" in all prose (code names
  `cash_*` unchanged).
- Added `docs/MODELS.md`, a comparison of every model with all hyperparameters, learning curves and
  held-out results.

**Validation**
- `ppo_junction` reached 7/7 doors in Gazebo with no collisions; returns within 0.5 of 2D.

**Task changes** (all models before this need retraining)
- Real LiDAR: the robot uses an LDROBOT D800 kit (STL-27L). Sim sensor now has 2160 points, 0.03–25 m
  and 1.5 cm noise. The policy sees ranges clipped to 10 m, the sensor's range on dark surfaces.
  Training drops 3% of beams to mimic glass.
- Slow reverse: max reverse 0.15 m/s (forward 0.5). Reversing fell from ~51% to ~9% of steps.
- Near-wall path cost: the path distance used by the reward ran 0.25 m from corners, inside the
  0.295 m collision boundary. Cells within 0.6 m of a wall now cost extra, so the route rounds
  corners at 0.46 m.

**Training** (PPO, 1.5M steps each, same hyperparameters as before)
- `ppo_d800` (real LiDAR + slow reverse): 200 held-out goals 93.0% success / 7.0% collisions
  (final model). 12 of 14 collisions clipped the junction's east corner.
- `ppo_d800_wallcost` (+ near-wall cost): 92.0% / 5.5% (best). East-corner hits fell from 12 to 3, but
  collisions moved to the west corner, always at full speed. Reached 90% eval at 0.4M steps
  (`ppo_junction`: 0.9M).
- `ppo_d800_prox` (+ proximity penalty: up to 0.5 per step when the footprint is within 0.3 m of an
  obstacle): **96.5% / 3.0% / 8.4% reversing** (best), matching `ppo_junction`'s 96% / 3% on the
  harder task. Corner hits fell from 9 to 3; the 6 left are 2 fountain, 3 corners and 1 end wall.
  Trained in 11 min with the CPU to itself. 100% on every periodic eval from 1.2M steps.
  Gazebo (STL-27L sensor model): 7/7 doors, no collisions, returns within 0.3 of 2D.
- Seeds 1 and 2 of `ppo_d800_prox`: across 3 seeds, **93.4 ± 2.8% success, 4.2 ± 1.5% collisions**,
  so seed 0 was lucky. Seed 2 mostly times out 0.4–1.1 m from the goal, moving back and forth.
  The "best" checkpoint (27-episode eval) is often worse than the final one.
- Released all three as `models-2026-10-15`.

**Safety layer** (`cash_rl/safety.py`, `evaluate --safety`)
- Simulates each command over the stopping time and falls back to gentler commands if the footprint
  would come within 10 cm of a scan point. On 3 seeds × 200 goals, collisions went from 4.2% to 0.7%;
  success went from 93.5% to 94.3%; it overrode 3.2% of steps. Collisions mostly became timeouts.
- Started a 3-seed run with 3M steps, linear LR decay (`--lr-schedule linear`) and a 100-goal
  periodic eval, but stopped it at 15% of seed 0 for time. To restart:
  `for s in 0 1 2; do ros2 run cash_rl train --name ppo_lr3m_s$s --seed $s --timesteps 3000000 --lr-schedule linear --eval-random 100 --goal-distance-range 1 25 --junction-prob 0.3; done`
  (~30 min per seed with nothing else running).

**Lessons**
- Don't run two trainings or several evaluations at once. PyTorch takes all 22 threads per process,
  and two runs stalled for 15 min. Use `OMP_NUM_THREADS=1` for parallel evaluations.
- The 27-episode periodic eval can't rank models (100% there, 90–93% on 200 goals).

**Open**
- Restart the 3M + LR-decay experiment (above). Then try the robot's measured velocity and 72
  beams as inputs, and a larger time penalty for the final-approach stalls (seed 2).
- Safety layer in Gazebo (`evaluate --env gazebo --safety`), then as a ROS node for the robot.
- Gazebo eval from varied start poses; 2–3 seeds per setup (cancelled today for time).
- LiDAR mounting position and height on the real chassis.

## 2026-10-14

**Mapping**
- Drove a full SLAM Toolbox pass and committed `maps/hallway.{pgm,yaml,posegraph,data}`
  (0.05 m/px, 30.1 × 12.55 m). SLAM started at the spawn pose, so map coordinates equal Gazebo
  world coordinates.
- Took three attempts. The first crashed into a wall and smeared the map. The second reused an
  orphaned `gz sim` server left running after Ctrl+C, so the map came out rotated ~40°.
  Fix: check `pgrep -af "gz sim"` before relaunching (now in README troubleshooting).

**Fast 2D training environment** (`cash_rl`)
- `HallwayRaycastEnv`: ROS-free simulator that raycasts 36 LiDAR beams against the SLAM map,
  with the Gazebo DiffDrive speed/acceleration limits. ~2,300 steps/s vs ~10 in Gazebo.
- `spec.py`: one shared definition of observation, action, collision and reward for the 2D env
  and the Gazebo `HallwayEnv`, so a trained policy runs in either unchanged.
- Validated against a live Gazebo `/scan`: 5.8 cm median beam error, from the map's 5 cm cells.
- Fixed a collision-check bug in `HallwayEnv`: the rear bumper could hit a wall without ending
  the episode.

**Task and reward**
- Goal-reaching task. Observation (41): 36 beams, goal distance and bearing, previous action.
- Reward: 10 × progress in geodesic (around-the-walls) goal distance − 0.01/step, +10 on reaching
  the goal (0.3 m), −20 on collision. Step limit scales with goal distance.
- Eval set: the 7 classroom doors from spawn plus 20 fixed random start/goal pairs.
- Added a sim-only PosePublisher to the URDF so `HallwayEnv` gets ground-truth pose
  (`/model/cash_bot/pose`); odometry drifted ~15 m in one mapping pass with no crash.

**Training** (PPO, Stable-Baselines3, CPU-only PyTorch; 12 parallel envs)
- `ppo_baseline`: 3M steps, ~38 min. Eval success 0% → 89% by 1M steps, best 93% at 2.8M.
  Reached 7/7 doors in Gazebo with no collisions; returns matched 2D within 0.2.
- Found the two branch doors flip-flopping between checkpoints; corridor doors were stable.
- `ppo_junction`: 30% of training episodes cross the junction, goals 1–25 m, 1.5M steps (~19 min).
  First reached 93% at 0.9M steps (baseline: 1.9M); branch doors reliable from 0.9M on.
- Head-to-head on 200 held-out random goals: `ppo_junction` 96.0% success / 3.0% collisions vs
  `ppo_baseline` 93.0% / 6.0%. Both reach all 7 doors from spawn. One seed per setup, so the gap
  isn't confirmed yet.

**Tools added**: `train`, `evaluate` (multi-model, held-out goals), `view_env` (watch or save
GIFs), `demo` (click to place the robot, right-click a destination).

**Open issues**
- From random headings both models drive in reverse ~51% of steps instead of turning around.
- The policy can hit the water fountain (x = 18 m) when it hugs the north wall heading for the
  branch.
- `ppo_junction` not yet tested in Gazebo; repeat each setup with 2–3 seeds.
- Next on the plan: random static obstacles, then pedestrians, then an unseen test layout.

## 2026-10-07

- Documented the WSL vs PowerShell mix-up in README troubleshooting.
- Planned the DRL approach: train in a custom fast 2D simulator built from the SLAM map (not
  Gazebo, ~10 steps/s), keep its interface identical to `HallwayEnv`, log periodic evaluation
  episodes for the report's learning curves, then validate in Gazebo before the Jetson.

## 2026-09-30

- Created the ROS 2 Jazzy workspace and `cash_sim` package: hallway world (30 m corridor with a
  branch, doors, bench, fountain, trash can), diff-drive robot with a 360° 2D LiDAR, sim launch
  with a headless option.
- Wired up SLAM Toolbox (fixed its lifecycle activation so it actually maps).
- Scaffolded `cash_rl`: Gymnasium `HallwayEnv` over ROS topics, service bridge for world reset
  and teleport, and a `random_agent` smoke test.
