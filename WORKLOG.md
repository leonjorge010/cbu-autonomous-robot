# Work log

What was done on each work day, newest first. Add a new `## YYYY-MM-DD` section at the top
for each session: what changed, results with numbers, and what's still open.

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
