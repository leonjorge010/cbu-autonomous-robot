# Work log

What was done on each work day, newest first. Add a new `## YYYY-MM-DD` section at the top
for each session: what changed, results with numbers, and what's still open.

## 2026-10-29

**Merged `pedestrians-no-yield` into `main`** (Ashlyn). Committed the map-generic TEGR code
(`14b094d`: `--map`, keepout zones, per-map pedestrian axes, limited Dijkstra), then merged the
branch. `main` now has non-yielding pedestrians, the lane penalty, `--encounter-prob`, the
privileged `avoider`, `dagger` and TEGR training. 63 tests pass. No new training or results.

**Open:** unchanged from 10-23 (DAgger seeds/rounds, TEGR wall collisions, `--init-from`,
subgoal agent).

## 2026-10-30

**Planner subgoals** (hierarchy step 1, Ashlyn). `HallwayRaycastEnv(subgoal_lookahead=M)` /
`--subgoal-lookahead M` on `train`, `dagger`, `evaluate`, `view_env`: the observed goal is the
point M m ahead on the geodesic route (`route(max_length=)`), the real goal once closer. Only
the observation changes. `evaluate`/`view_env` default to the model's training setting. 2 new
tests; 65 pass. No training run.

Existing models, TEGR, 200 held-out goals, no subgoal / 3 m / 5 m (files
`runs/eval_tegr_subgoal*.txt`, `runs/eval_tegr_zeroshot_subgoal*.txt`):

| Model | Test | none | 3 m | 5 m |
|---|---|---|---|---|
| `tegr_ppo` final | no pedestrians | 86.5% | 85.5% | 85.0% |
| `tegr_dagger_p1` | encounter, 1 ped | 61.5% | 55.5% | 61.5% |
| `tegr_ppo` final | encounter, 1 ped | 42.0% | 43.0% | 43.0% |
| `ppo_d800_prox` final (hallway, zero-shot) | no pedestrians | 60.0% (34% timeout) | 64.0% (30.5%) | 63.5% (30.5%) |

- No real change for models trained without subgoals: TEGR models barely time out (0–2%), so
  their failures are collisions, which subgoals don't touch.
- The hallway model's zero-shot timeouts barely move (34% → 30.5%), so they are not mainly
  the straight-line goal; cause still unknown (it reverses on ~18% of steps).
- The real test is a model *trained* with subgoals: hallway-trained → TEGR zero-shot (unseen map)
  and TEGR DAgger with subgoals.

## 2026-10-23

**Training on the real TEGR 2nd floor** (`cbu_floor2`, Ashlyn; schedule weeks 8–9). Branch
`pedestrians-no-yield`; the map-generic code below is **uncommitted** (last commit `eead4e9`).
- `--map cbu_floor2` (name in `maps/` or a path) on `train`, `dagger`, `evaluate`, `view_env`;
  `train`/`dagger` record `map_yaml` in `config.json`, and `evaluate`/`view_env` default to the
  model's map (warning on a mismatch). Hallway-only: door evals (`env.door_goals` is empty
  elsewhere, so `evaluate` needs `--random N`) and `--junction-prob` (raises on other maps).
- `<map>_keepout.yaml` keeps spawns, goals and pedestrians out of Stair 1 and the learning zones
  (the robot can still drive in: nothing penalizes it there).
- Pedestrians walk along the longer open stretch through their cell, majority-voted over 1.5 m
  (before: y if y ≥ 1.5 m, so every TEGR pedestrian walked north–south). The hallway keeps its old
  rule: hallway seeded rollouts verified bit-identical before/after.
- Spawn-side distance field stops at the goal range (`distance_field(limit=)`): TEGR reset
  213 → 111 ms. `view_env` labels the layout's room goals (`landmarks_for`) and fits the map.
- Room-goal check: `trace_floorplan --overlay` + a labelled render; all 26 look right, except
  `elevator_2` (43.0, 57.4) may sit ~2 m east of the elevator door (check on site).
- 7 new tests; 63 pass.

**Overnight queue** (`runs/queue_tegr.txt`, all 10 jobs OK, 01:52–10:56). 200 held-out goals
(1–15 m) on TEGR; files `runs/eval_tegr_*.txt`.

| Encounter test, 1 pedestrian | Success | Collision | Reversing |
|---|---|---|---|
| avoider (keep-right teacher) | 89.0% | 10.5% | 5.2% |
| **`tegr_dagger_p1`** | **61.5%** | 38.5% | 6.1% |
| `tegr_ppo_p1` (enc 0.7, no lane, ent 0.001) | 52.0% | 45.5% | 21.4% |
| `downhill` | 50.5% | 49.5% | 0% |
| `tegr_ppo` (no pedestrians) | 42.0% | 56.5% | 13.1% |

- First learned model clearly above `downhill` (+11 points, one seed). DAgger per-round eval was
  still rising at round 10 (44% → 57% → 61% at rounds 5, 8, 10); imitation MSE 0.115.
- No pedestrians: `tegr_ppo` seeds 0/1/2 final 86.5/79.5/87.0% (mean 84.3%), best checkpoints
  87.0/89.5/89.0% (mean 88.5%); `downhill` 100%. Failures are mostly wall collisions (10–14%),
  vs 96.5% for `ppo_d800_prox` on the hallway. Hallway models zero-shot on TEGR: 57–60% (~35%
  timeouts).
- 4 random pedestrians: `tegr_ppo_p4` 78.0%, `tegr_dagger_p1` 76.5%, `downhill` 91.5%, avoider
  96.0%. Random pedestrians rarely meet the robot here, so this gap is the wall-collision weakness.
- The student still almost never swerves hard near a pedestrian: 5% of steps vs the teacher's 43%
  (hallway student before keep-right: 13%). Its gain did not come from fixing the left/right
  averaging.
- Reports: progress update https://claude.ai/artifact/BiPTYtyuFbgmBCrZxjjmK3 and results-with-charts
  doc https://claude.ai/artifact/VsfudV5hPgKR2pYEggH6VL (both private until shared). Representative
  demo: `view_env --pedestrians 1 --encounter-prob 1 --seed 1100000 --episodes 10` for `downhill`
  (`--map cbu_floor2`), `tegr_ppo_p1`, `tegr_dagger_p1` (4/6/7 of 10).

**Open**
- Commit the map-generic code.
- DAgger: 2 more seeds and more rounds (`--rounds 20` or `--steps-per-round 120000`) to confirm
  61.5% and find the plateau.
- `--init-from` warm start for `train`/`dagger` (postponed): PPO fine-tune from the student.
- Student that picks among the teacher's 54 discrete actions, if averaging stays the limit.
- TEGR wall collisions: longer training and/or `evaluate --safety`.
- Keep-right teacher with 4 pedestrians drops to 78% on the hallway (84.5% without the rule).
- Long routes on TEGR need subgoals (goal obs is straight-line): the planned subgoal agent.
- Hallway `dagger_p1_kr` rerun never done (superseded by TEGR). `demo` still assumes hallway doors.
- `runs/queue.txt` still lists 4 stale ent-0.01 runs.

## 2026-10-22

**Encounters, privileged avoider, teacher-student training** (hallway, 1 pedestrian, Ashlyn).
- `HallwayRaycastEnv(encounter_prob)` / `--encounter-prob`: the first pedestrian starts 4–10 m
  ahead on the robot's route walking at it. The *encounter test* (`--encounter-prob 1`) separates
  avoidance far better than random starts.
- `scripted.avoider`: privileged upper bound (true future pedestrian positions). Limited to
  pedestrians visible in the 36 beams it scores the same, so the observation is sufficient; the
  problem is learning. Fixed a bug where it froze at rest facing its goal (3/7 doors → 7/7;
  encounter test 84.5% → 88.5%). Added keep-right (`side_weight`, judged in the pedestrian's
  walking frame): keeps right on 72% of close passes (was 53%), encounter 86.5%, 4 peds 78.0%.
- `train --ent-coef`: 0.01 blew the action std up to 4.6; 0.001 is stable but changed nothing.
- Sweep (1 ped, standard test): lane penalty + encounters 60.0% / 22.8% reversing; lane penalty off
  73–79.5% / ~11%; `downhill` 85%, avoider 94%. Encounter test: PPO 35.5–41%, `downhill` 43.5%.
- `dagger` (new console script, `eead4e9`): DAgger with the avoider as teacher, student = PPO
  MlpPolicy, saves PPO zips. `dagger_p1`: 44.5% on the encounter test (`downhill` 43.5%);
  behavior cloning alone 8.5%. Near a pedestrian the teacher swerves hard on 43% of steps split
  left/right; the student averages them to straight (13%).

## 2026-10-21

**Pedestrians that don't yield; lane penalty** (`HallwayRaycastEnv`, Ashlyn; schedule week 8). Branch
`pedestrians-no-yield`, not merged yet.
- `ppo_peds` had learned to rely on pedestrians stepping around it: against pedestrians that walk
  straight on, its success fell from 90.5% to 36% (100 held-out goals, 4 pedestrians).
- Pedestrians now ignore the robot and walk into it if it stays in their path; the old behavior is
  `pedestrians_yield=True` / `--yielding-pedestrians` (`train`, `evaluate`, `view_env`, `demo`),
  to reproduce runs from before today. Speed lowered from 0.8–1.4 to 0.5–0.9 m/s.
- Waiting is free: stopped (< 0.05 m/s) with a pedestrian within 0.8 m, the robot pays no time or
  pedestrian proximity penalty (`spec.waiting_for_pedestrian`). Being walked into still costs −40.
- Lane penalty (`PEDESTRIAN_LANE_*`, `spec.pedestrian_lane_penalty`): up to −0.3 per pedestrian per
  step while the footprint is in a pedestrian's lane and they would reach it within 3 s at their
  closing speed, waiting or not. Reason: the 0.8 m proximity penalty starts only 0.6–0.8 s before
  contact, too late for a diff-drive robot to turn and get clear. Costs ~10% env speed.
- `demo --pedestrians N` (new pedestrians placed at the start of each leg).

**Results** (200 held-out goals, 4 pedestrians, one seed each)

| Model | Success | Collision | Timeout | Reversing |
|---|---|---|---|---|
| `ppo_peds_noyield` (3M) | 48.0% | 50.0% | 2.0% | 14.5% |
| `ppo_peds_lane` final (6M, lane penalty) | 28.5% | 61.0% | 10.5% | 28.1% |
| `ppo_peds_lane` best_model | 51.5% | 48.0% | 0.5% | 14.1% |

- Neither model avoids pedestrians: `downhill`, which ignores them entirely, gets 57% success / 43%
  pedestrian collisions on the same episodes. Most collisions are the robot driving into a
  pedestrian walking toward it (front or side), mid-episode; few happen while it is stopped.
- The lane penalty taught backing away (reverse doubled; 0.15 m/s reverse can't outrun a
  pedestrian) instead of stepping aside.
- On the TEGR map (`CASH_MAP_YAML=maps/cbu_floor2.yaml`) `ppo_peds_noyield` reached 6/30 random
  goals, mostly timing out. `view_env`/`demo` still draw the hallway's door labels there (they sit
  outside the building), and pedestrians assume the hallway's corridor directions.

**Open (pedestrians)**
- Find out whether avoidance is possible at all here (36-beam view sees a pedestrian reliably only
  within ~2.9 m; getting clear takes ~2.5–3 s) or PPO just doesn't find it (`ent_coef` is 0):
  1. curriculum: `train --pedestrians 1 ...` — does it beat `downhill` with 1 pedestrian?
  2. a scripted avoider that knows the pedestrians' positions, as an upper bound.
- Then revert or rework the lane penalty, or try an entropy bonus — one change at a time.
- Seed-1 repeat of `ppo_peds_noyield` was queued but never started.

## 2026-10-16

**Moving pedestrians and scan stacking** (`HallwayRaycastEnv`, Ashlyn; schedule week 8)
- `HallwayRaycastEnv(n_pedestrians=N)`, default 0 so existing runs are unchanged: 0.25 m discs
  walking the corridor axis at 0.8–1.4 m/s, turning back only at walls. They block beams (exact
  ray/disc intersection, so the safety layer sees them too) and count as collisions. 2D only;
  Gazebo has no pedestrians yet.
- Pedestrians never turn back for the robot: with it ahead inside a 0.8 m gap they slow to 30%; if
  it is in their path they sidestep away at 0.5 m/s (stopping inside 0.3 m until clear), then keep
  walking the same way. A parked robot is never hit (0 hits, 100 episodes × 6 pedestrians).
- Reward: up to −0.3 per pedestrian per step inside the 0.8 m gap, ramping to contact; hitting a
  pedestrian costs −40 instead of a wall's −20. Step info adds `hit_pedestrian`,
  `min_pedestrian_gap`.
- Observation stacks the last 3 scans (41 → 113 values) so the policy can see motion; `HallwayEnv`
  builds the same stack. 41-value models (all models so far, incl. `ppo_d800_prox`) still load:
  `train.load_model` feeds them the newest scan only.
- `train --pedestrians N` (training *and* periodic eval, so compare runs at equal counts),
  `evaluate --pedestrians N`, `view_env --pedestrians N` (purple discs).
- `downhill`, which ignores pedestrians, with 4 of them: 82/100 goals, 18 pedestrian hits.
  Training speed with 4 pedestrians: 790 vs 855 steps/s without (~63 min per 3M steps measured
  today, slower than the ~38 min seen 10-14).
- Merged with this day's `main` (STL-27L LiDAR, wall proximity penalty, safety layer): pedestrian
  and wall penalties apply side by side; `evaluate --pedestrians N --safety` works.

**Open (pedestrians)**
- Train a model on the new pedestrian env, e.g. `train --pedestrians 4 --goal-distance-range 1 25
  --junction-prob 0.3 --timesteps 3000000 --name ppo_peds`, and compare against the 41-value
  baselines with `evaluate --random 200 --pedestrians 4`.
- Later: add a constant student following the robot, to model the user being escorted (keep them
  within following distance; penalize losing them).

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
- `sim.launch.py` has a `world` argument: `world:=cbu_floor2 x:=14.1 y:=30 yaw:=1.57` spawns in the
  west corridor.
- Checked with the existing code (`CASH_MAP_YAML=maps/cbu_floor2.yaml`): one connected traversable
  area, every door ≥ 0.65 m from walls, geodesic field 36 ms/goal, `downhill` reaches 40/40 random
  goals and crosses the building (west corridor → 221/205/elevator_2) with no collisions.

**Open**
- Not checked in Gazebo yet (needs a run: `ros2 launch cash_sim sim.launch.py world:=cbu_floor2
  gui:=true rviz:=false x:=14.1 y:=30 yaw:=1.57`).
- In person: tape one corridor width (fixes the ±2–5% photo scale); what the atrium edge is (glass
  railing would give no LiDAR returns); whether TEGR 214/216 have corridor doors (their main doors
  face Learning Zone 3, so their goals sit at the zone opening); restroom door positions; what the
  "D2-x" wall items are.
- Env support for this map: keep-out zones in spawn/goal sampling and the geodesic (a virtual wall),
  per-map door goals and spawn instead of `spec.DOOR_GOALS`/`SPAWN_POSE`, and the hallway-specific
  junction sampling (`BRANCH_MIN_Y`). Goals can now be up to ~85 m away, past
  `MAX_GOAL_DISTANCE` (30 m), so long escorts should be chained through waypoints.

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
