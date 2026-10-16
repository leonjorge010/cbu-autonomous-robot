# Work log

What was done on each work day, newest first. Add a new `## YYYY-MM-DD` section at the top
for each session: what changed, results with numbers, and what's still open.

## 2026-10-16

**Real floor plan → sim world** (TEGR 2nd floor, from a phone photo of the printed plan, `docs/cbu_floorplan.jpg`)
- `src/cash_sim/tools/trace_floorplan.py`: perspective-corrects the photo from the building's 4 outer
  corners, scales it with the plan's 0–40 ft bar (40 ft = 415 px, 2.94 cm/px; building 61 × 72 m),
  extracts the thick wall lines, and keeps the walkable area inside hand-placed corridor rectangles
  whose edges sit on the wall lines, so every doorway is closed. Label text is removed; three door
  leaves drawn swung open into the corridor are cleared (modeled shut).
- Measured corridor widths: south (201–205) ~2.3 m, west (228–230) ~3.0–3.2 m, north (221–226) ~2.3 m.
  Narrower than the 3 m guessed by eye; the sim hallway is 2.4 m.
- Result `src/cash_sim/worlds/cbu_floor2_layout.yaml`: 594 m² of free space (corridors, Gallery 2,
  women's restroom vestibule, alcove outside 228, Elevator 2 lobby), the counter in Learning Zone 3,
  keep-out rectangles for Learning Zones 2/3/4 (open to the corridor, so traced as open space), and
  24 door goals on the corridor centerline facing each door.
- Fixed the Gallery 2 corner (spotted against the plan): Stair 1 has no wall on the gallery side, so
  its flights are now open space + keep-out (one flight goes down; a 2D LiDAR can't see a drop, so the
  real robot must rely on the keep-out), the stair core is an obstacle, and `elevator_1`/`stair_1`
  goals were added (26 total). Elevator 1's door faces the west end of the 201–205 corridor.
- `src/cash_sim/tools/build_floorplan_world.py` turns the layout into `worlds/cbu_floor2.sdf`
  (329 wall boxes, `gz sdf -k` valid), `maps/cbu_floor2.{pgm,yaml}` and the Nav2 keepout mask
  `maps/cbu_floor2_keepout.{pgm,yaml}`. Wall boxes vs. traced polygon: 20 of 143k sample points
  overlap (sharp corners); the map matches to within one 5 cm cell.
- `sim.launch.py` has a `world` argument: `world:=cbu_floor2 x:=9.6 y:=16.8 yaw:=0` spawns at the
  default start (see below).
- Checked with the existing code (`CASH_MAP_YAML=maps/cbu_floor2.yaml`): one connected traversable
  area, every door ≥ 0.65 m from walls, geodesic field 36 ms/goal, `downhill` reaches 40/40 random
  goals and crosses the building (west corridor → 221/205/elevator_2) with no collisions.

**2D env support for any map** (`cash_rl`)
- Per-map sidecars next to `MAP.yaml`: `MAP_goals.yaml` (spawn, named goals; `branch_min_y` for the
  hallway's `--junction-prob`) and `MAP_keepout.yaml`. The hallway's goals moved out of `spec.py`
  into `maps/hallway_goals.yaml`; `build_floorplan_world.py` writes `cbu_floor2_goals.yaml`.
- Keep-out = virtual wall: never spawned in or used as a goal, not traversable for the geodesic, and
  entering it with the robot's center ends the episode as a collision (`info['keepout']`). Same in
  `HallwayEnv` (Gazebo), whose world name now follows the map (`train_env.launch.py world:=...`).
- `lookahead`: the policy observes the route waypoint that far ahead instead of the goal, so 85 m
  escorts stay inside the 30 m observation range. `train`/`evaluate`/`view_env`/`demo` take `--map`
  and `--lookahead`; `config.json` records the map, and evaluate/view_env/demo default to it.
- Tests: 43 pass, incl. new `test_floor_plan_map.py` (keep-out termination, sampling, waypoints,
  far goals reachable). Old hallway model unchanged: `ppo_d800_prox` 7/7 doors, 98% on 50 held-out.
- Zero-shot `ppo_d800_prox` on `cbu_floor2` (lookahead 10 m): 68% of 100 held-out random goals
  (14% keep-out, 2% wall, 16% timeout); 5/26 named goals from the spawn. Most north-bound runs clip
  the Learning Zone 3 counter at the zone's corridor opening (15.6, 36.3). It handles the real
  corridors but has never been penalized for keep-out, which its LiDAR can't see.


**Start pose, launch and RViz**
- Default start on `cbu_floor2` is the `stair_1` stop in Gallery 2, facing east (9.6, 16.8, 0) — the
  robot's starting point for now. Named-goal evals skip goals at the start pose.
- `sim.launch.py`/`train_env.launch.py`: unset `x`/`y`/`yaw` take the world's spawn from its goals
  file, so `world:=cbu_floor2` alone starts at `stair_1`.
- RViz: `sim.launch.py map:=cbu_floor2` adds `map_view.launch.py` — `nav2_map_server` for `/map` and
  `/keepout_mask`, `goal_markers` (labeled arrows per goal, green spawn arrow) and a static map→odom
  equal to the spawn pose (no localization; odometry drift uncorrected; not with SLAM). Checked
  headless on a separate ROS domain; not yet looked at in RViz.
- Lookahead waypoints use a per-goal downhill-pointer array instead of a 3×3 search per cell:
  env with lookahead 978 → 1455 steps/s (2002 without).

**First policy on the real floor:** `ppo_floor2_s0` (PPO, 3M steps, 45 min, `--lookahead 10
  --goal-distance-range 1 25`). Periodic eval swung 76% → 42% → 91% (best, 2.75M). Held-out 200:
  best 90.0% success / 9% wall / 1% keep-out (zero-shot hallway model: 60.5% / 5.5% / 16%),
  23/25 named goals; with the safety layer 92.0% / 0% wall / 1% keep-out / 7% timeout, 24/25.
  Wall hits are inside corners at full speed (Gallery 2 → south corridor at the men's restroom
  block, west → north corridor). `evaluate` now reports keep-out entries separately. Details in
  `docs/MODELS.md`. Model weights are local only (`runs/`, not released yet).

**Next** (paused here; in suggested order)
1. Seeds 1 and 2 of `ppo_floor2` (one at a time, ~45 min each; same command with `--seed 1/2`,
   see `docs/MODELS.md` → Reproducing). Consider `--lr-schedule linear` for the eval swings. Then
   release the weights like `models-2026-10-15`.
2. Gazebo check of this world: `ros2 launch cash_rl train_env.launch.py world:=cbu_floor2`, then
   `CASH_MAP_YAML=~/ros2_ws/maps/cbu_floor2.yaml ros2 run cash_rl evaluate
   runs/ppo_floor2_s0/best_model.zip --env gazebo` (training stopped). Also look at the world in
   the Gazebo GUI and the `map:=cbu_floor2` RViz view.
3. Escort node in ROS: localize on `cbu_floor2` (Nav2 AMCL), "escort to TEGR X" → route →
   10 m waypoints → numpy-exported policy → safety layer + keep-out check → `/cmd_vel`. Test end to
   end in Gazebo; the same node goes on the Jetson (milestones 2, 4 and the numpy export).
4. Inside-corner crashes: more corner-crossing training episodes (generalize `--junction-prob`
   beyond the hallway) or a stronger near-wall route cost, so the safety layer has less to catch.
5. Obstacles, then pedestrians, in the 2D env (main open item for the CSCI 4220 report).
6. In person: tape one corridor width (fixes the ±2–5% photo scale); what the atrium edge is (glass
   railing would give no LiDAR returns); whether TEGR 214/216 have corridor doors (their main doors
   face Learning Zone 3, so their goals sit at the zone opening); restroom door positions; what the
   "D2-x" wall items are; whether Stair 1 has a railing or gate.

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
- Gazebo (2160-beam scan): 7/7 doors with no interventions. Driving full speed at the end wall, it
  stopped 10.2 cm and 10.4 cm from the wall; without it, both runs collided.
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
- Safety layer as a ROS node between the policy and `/cmd_vel` for the robot.
- Generalize beyond this one hallway. The sim hallway is invented, so the real CBU building is
  itself a new map.
  - Carries over unchanged: the sensor model, reward, PPO settings, tools and the safety layer.
  - Per new map: a SLAM map (~20–30 min drive), `CASH_MAP_YAML`, goal/door list, then retrain
    (3 seeds) and evaluate.
  - One-time code work: the junction-crossing episodes (`BRANCH_MIN_Y`, junction at x = 14 m),
    `DOOR_GOALS`/`SPAWN_POSE` and some tests assume this map's layout.
  - The policy only sees LiDAR + relative goal, but has trained on one map, so it's untested on a
    layout it has never seen. Plan, after random obstacles: train on several maps or procedurally
    generated hallways (varied widths, junctions, obstacles), and hold out one layout as the test.
  - Deployment still needs a map for localization and room waypoints (milestone 2). The policy does
    local driving to waypoints and the safety layer protects at the last moment.
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
