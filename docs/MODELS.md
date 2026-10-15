# Model comparison

Every navigation policy trained so far: what task each one learned, how it was trained, and how it
scored. All numbers below were measured on 2026-10-15. Each model was evaluated with the code it
was trained on, so each row is scored on its own task.

**Current recommendation: `ppo_d800_prox`** (seed 0, best checkpoint). It matches the code on `main`
(real LiDAR model, slow reverse, near-wall cost, proximity penalty). Of the five seed-0 models it
ties for the best success and collision rates, while reversing 8% of the time instead of 51%. Over
three training seeds the same setup averages **93.4% success and 4.2% collisions** (see
[Seed variation](#seed-variation)). Download it from the
[`models-2026-10-15` release](https://github.com/leonjorge010/cbu-autonomous-robot/releases/tag/models-2026-10-15).

## Summary

Held-out test: 200 random start/goal pairs the policy never saw in training or in periodic
evaluation (random start heading, goals 1–25 m away along the hallway).

| Model | Task | Success | Collisions | Timeouts | Steps reversing | All 7 doors |
|---|---|---|---|---|---|---|
| `ppo_baseline` | old sensor, fast reverse | 93.0% | 6.0% | 1.0% | 51.9% | ✅ (also in Gazebo) |
| `ppo_junction` | old sensor, fast reverse | **96.0%** | **3.0%** | 1.0% | 50.5% | ✅ (also in Gazebo) |
| `ppo_d800` (best) | real LiDAR, slow reverse | 90.5% | 8.0% | 1.5% | 8.0% | ✅ |
| `ppo_d800` (final) | real LiDAR, slow reverse | 93.0% | 7.0% | 0.0% | 8.6% | ✅ |
| `ppo_d800_wallcost` (best) | + corner-safe route | 92.0% | 5.5% | 2.5% | 10.7% | ✅ |
| `ppo_d800_wallcost` (final) | + corner-safe route | 91.5% | 8.5% | 0.0% | 9.2% | ✅ |
| `ppo_d800_prox` (best) ⭐ | + proximity penalty | **96.5%** | **3.0%** | 0.5% | 8.4% | ✅ (also in Gazebo) |
| `ppo_d800_prox` (final) | + proximity penalty | 96.0% | **3.0%** | 1.0% | **8.2%** | ✅ |

The newer models do a slightly harder task (noisy real sensor, slow reverse), so the two groups
are not directly comparable. What changed:
- **Reversing dropped from ~51% to ~8% of steps.** The old models drove backward half the time
  instead of turning around.
- **Learning got faster:** 90% eval success at 0.1–0.6M steps vs 0.9–1.9M.
- **Collisions rose with the real sensor, then came back down.** They went from the old 3% to
  5.5–8.5%, and the proximity penalty brought them back to 3%. See
  [Collision analysis](#collision-analysis).

## The task each model learned

Every model drives from a start pose to a goal point using only LiDAR and the goal's relative
position. Settings that differ between models are in bold.

| Setting | `ppo_baseline` | `ppo_junction` | `ppo_d800` | `ppo_d800_wallcost` | `ppo_d800_prox` |
|---|---|---|---|---|---|
| LiDAR model | ideal | ideal | **LDROBOT STL-27L** | STL-27L | STL-27L |
| Scan points / range | 360 / 0.15–12 m | 360 / 0.15–12 m | **2160 / 0.03–25 m** | 2160 / 0.03–25 m | 2160 / 0.03–25 m |
| Range the policy sees | 12 m | 12 m | **10 m** | 10 m | 10 m |
| Range noise (std dev) | 1.0 cm | 1.0 cm | **1.5 cm** | 1.5 cm | 1.5 cm |
| Missing returns (training) | none | none | **3% of beams** | 3% of beams | 3% of beams |
| Max forward / reverse speed | 0.5 / 0.5 m/s | 0.5 / 0.5 m/s | 0.5 / **0.15** m/s | 0.5 / 0.15 m/s | 0.5 / 0.15 m/s |
| Training goal distance | 1–15 m | **1–25 m** | 1–25 m | 1–25 m | 1–25 m |
| Junction-crossing episodes | 0% | **30%** | 30% | 30% | 30% |
| Near-wall path cost | none | none | none | **within 0.6 m of a wall** | within 0.6 m of a wall |
| Proximity penalty | none | none | none | none | **up to 0.5/step within 0.3 m** |

**Same for every model**
- **Observation (41 values):** 36 LiDAR distances, one every 10°, divided by the max range. Goal
  distance / 30 m, cos and sin of the goal bearing, and the previous action.
- **Action (2 values in [-1, 1]):** forward speed and turn rate (max 1.0 rad/s), sent every 0.1 s.
  Acceleration limits are 1.0 m/s² and 2.0 rad/s².
- **Reward per step:** 10 × (decrease in path distance to the goal) − 0.01. Reaching the goal
  (within 0.3 m) gives +10. A collision gives −20. Both end the episode.
  - The path distance goes *around* walls, not straight through them.
  - From `ppo_d800_wallcost` on, it charges extra near walls: 1 + 8 × (1 − c / 0.6)² per meter for a
    point c meters from a wall, which is 3× at the collision boundary. The cheapest route then
    rounds corners 0.46 m from the wall instead of 0.25 m.
  - `ppo_d800_prox` also subtracts 0.5 × (1 − c / 0.3)² each step that the gap c between the robot's
    edge and the nearest obstacle is under 0.3 m. That's 0.13 at 0.15 m and 0.35 at the 5 cm
    collision margin. For comparison, driving forward at full speed earns about 0.5 per step.
- **Collision:** anything within 5 cm of the 0.50 × 0.49 m robot footprint.
- **Episode time limit:** 2 × the time to drive the path distance at full speed, plus 10 s.

## Training setup (identical for all five)

| | |
|---|---|
| Algorithm | PPO ([Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 2.9.0, PyTorch 2.14 CPU) |
| Network | Separate actor and critic MLPs, each 41 → 256 → 256 (tanh), Gaussian action head |
| Parameters | 153,861 total; **77,060 in the actor** (the part that runs on the robot) |
| Parallel environments | 12 (`SubprocVecEnv`) |
| Rollout per update | 512 steps × 12 envs = 6,144 transitions |
| Minibatch / epochs | 256 / 10 (240 gradient steps per update) |
| Learning rate | 3e-4 (constant) |
| Discount γ / GAE λ | 0.99 / 0.95 |
| Clip range / entropy coef. | 0.2 / 0.0 |
| Seed | 0 (one seed per setup) |
| Simulator | 2D raycaster on the SLAM map, ~2,300 steps/s per core |
| Periodic evaluation | every 100k steps, deterministic: 7 doors from spawn + 20 fixed random pairs = 27 episodes. `best_model.zip` is the checkpoint with the highest success rate on these |

## Training results

| Model | Steps | Wall time | First ≥90% eval | Best eval (27 episodes) |
|---|---|---|---|---|
| `ppo_baseline` | 3.0M | 36 min | 1.9M | 93% at 1.9M |
| `ppo_junction` | 1.5M | 14 min | 0.9M | 96% at 1.1M |
| `ppo_d800` | 1.5M | 27 min | 0.6M | 100% at 0.6M |
| `ppo_d800_wallcost` | 1.5M | 26 min | 0.4M | 100% at 0.9M |
| `ppo_d800_prox` | 1.5M | **11 min** | **0.1M** | 100% at 1.2M (and every eval after) |

Wall times are on an RTX 4060 laptop (CPU only, 22 threads). `ppo_d800` and `ppo_d800_wallcost` were
slowed by other jobs sharing the CPU. `ppo_d800_prox` had the machine to itself, which is what a
1.5M-step run normally takes.

![Learning curves](learning_curves.png)

**The 27-episode evaluation is too small to rank models.** All three D800 runs hit 100% there but
scored 90–96.5% on the 200-goal test. Use the curves to see learning speed and stability, and the 200-goal
test to compare final quality.

## Detailed held-out results

200 held-out random pairs (2D). Time and path are averaged over successful episodes only. Path is
the distance driven ÷ the shortest path distance at the start.

| Model | Success | Collision | Timeout | Avg. time | Path | Reversing | Return |
|---|---|---|---|---|---|---|---|
| `ppo_baseline` | 93.0% | 6.0% | 1.0% | 15.3 s | 1.01 | 51.9% | +74.5 |
| `ppo_junction` | 96.0% | 3.0% | 1.0% | 15.8 s | 1.03 | 50.5% | +77.6 |
| `ppo_d800` best | 90.5% | 8.0% | 1.5% | 16.5 s | 1.05 | 8.0% | +72.2 |
| `ppo_d800` final | 93.0% | 7.0% | 0.0% | 16.5 s | 1.04 | 8.6% | +74.2 |
| `ppo_d800_wallcost` best | 92.0% | 5.5% | 2.5% | 17.4 s | 1.03 | 10.7% | +76.1 |
| `ppo_d800_wallcost` final | 91.5% | 8.5% | 0.0% | 16.6 s | 1.00 | 9.2% | +74.1 |
| `ppo_d800_prox` best | 96.5% | 3.0% | 0.5% | 16.3 s | 0.99 | 8.4% | +78.8 |
| `ppo_d800_prox` final | 96.0% | 3.0% | 1.0% | 16.2 s | 0.98 | 8.2% | +78.8 |

Returns from `ppo_d800_wallcost` and `ppo_d800_prox` use different rewards, so returns can't be
compared across those rows.

**Doors from spawn:** every model reaches all 7 classroom doors with no collisions, in 29.4–30.5 s
on average. **Gazebo:** `ppo_baseline`, `ppo_junction` and `ppo_d800_prox` (best) each reached 7/7
doors in Gazebo with no collisions, and their returns were within 0.5 of 2D. For `ppo_d800_prox`,
Gazebo used the STL-27L sensor model (2160 beams, 0.03–25 m), and episodes took 1–11 fewer steps
than in 2D. `ppo_d800` and `ppo_d800_wallcost` have not been run in Gazebo.

## Seed variation

`ppo_d800_prox` was trained three times with identical settings and seeds 0, 1 and 2. Each row is
200 held-out goals.

| Seed | Checkpoint | Success | Collisions | Timeouts | Reversing |
|---|---|---|---|---|---|
| 0 | best | 96.5% | 3.0% | 0.5% | 8.4% |
| 0 | final | 96.0% | 3.0% | 1.0% | 8.2% |
| 1 | best | 91.5% | 6.0% | 2.5% | 7.9% |
| 1 | final | 94.0% | 5.5% | 0.5% | 8.6% |
| 2 | best | 92.0% | 3.5% | 4.5% | 12.2% |
| 2 | final | 90.5% | 4.0% | 5.5% | 16.4% |
| **Mean ± std** | best | **93.3 ± 2.8%** | **4.2 ± 1.6%** | 2.5 ± 2.0% | 9.5 ± 2.4% |
| **Mean ± std** | final | **93.5 ± 2.8%** | **4.2 ± 1.3%** | 2.3 ± 2.8% | 11.1 ± 4.6% |

All three seeds reach all 7 doors from spawn.

- **Seed 0 was the luckiest run.** Its 96.5% / 3.0% is about one standard deviation above the
  mean, so compare setups by the mean, not by seed 0.
- **One-seed comparisons above are within this noise.** The ±3% spread in success is as large as
  most of the differences between the seed-0 models in the Summary.
- **Seed 2 fails differently.** 10 of its 11 timeouts end 0.4–1.1 m from the goal, with the robot
  moving back and forth in place. It reaches the goal area and then can't finish the last
  approach. That habit is also where its extra reversing comes from. Seeds 0 and 1 rarely do this.
- **"Best" checkpoints aren't reliably better** than the end-of-training model. They're picked on 27
  episodes, which is too few to pick between checkpoints.

## Collision analysis

Where the 2D collisions happen on the 200-goal test:

| Model | Collisions | East corner of the junction (x ≈ 15) | West corner (x ≈ 13) | Elsewhere |
|---|---|---|---|---|
| `ppo_d800` final | 14 | **12** | 1 | 1 (fountain area, x ≈ 18) |
| `ppo_d800_wallcost` best | 11 | 3 | **6** | 2 (fountain area, x ≈ 18) |
| `ppo_d800_prox` best | 6 | 1 | 2 | 2 (fountain area), 1 (end wall, started 0.47 m away) |

- **In `ppo_d800`, the reward itself caused the collisions.** The route the reward follows passed
  0.25 m from the corners, but anything closer than 0.295 m counts as a collision. The near-wall
  path cost fixed that, and east-corner hits fell from 12 to 3.
- **The collisions then moved to the west corner.** Most come from episodes that start inside the
  junction facing a random direction. The robot is nearly always at full speed (0.5 m/s) when it
  hits. At that speed its tightest turn has a 0.5 m radius, so it swings wide into the wall.
- **The proximity penalty supplied the missing signal.** Nothing in the earlier rewards said "slow
  down near walls." The penalty brought corner hits from 9 to 3 and total collisions from 11 to 6.
  Most of what's left comes from awkward starts, such as inside the junction or facing a wall.
- **The water fountain** sticks out into the corridor and now causes as many hits as the corners.
  Training with random obstacles, the next planned step, should help with it.

## Caveats

- **Only `ppo_d800_prox` has multiple seeds.** It spreads ±2.8% in success and ±1.6% in
  collisions across seeds, and the other setups are probably similar. Their single-seed differences
  of a few percent aren't meaningful.
- **Gazebo covers only the 7 doors from spawn.** Random start poses aren't supported there yet.
- **The proximity penalty in Gazebo** measures distance from the scan, while 2D uses the map. They
  agree to about one map cell (5 cm), and the policy never sees the penalty at run time.
- **Every model trained on one hallway map** with no people or moving obstacles. These numbers
  don't show how a policy handles a new building.

## Reproducing

Commands are run from `~/ros2_ws`. Each run's `config.json` (included in the releases) records
every setting.

| Model | Code | Train command |
|---|---|---|
| `ppo_baseline` | `5cb4ade` | `ros2 run cash_rl train --name ppo_baseline --timesteps 3000000` |
| `ppo_junction` | `5cb4ade` | `ros2 run cash_rl train --name ppo_junction --timesteps 1500000 --goal-distance-range 1 25 --junction-prob 0.3` |
| `ppo_d800` | `14351b5` | same as `ppo_junction` |
| `ppo_d800_wallcost` | `b333324` to `4081d99` | same as `ppo_junction` |
| `ppo_d800_prox` | `5f4369e` or later | same as `ppo_junction` |

Evaluate any model with
`ros2 run cash_rl evaluate runs/<name>/best_model.zip --random 200`. Run the old models' code
(`git checkout 5cb4ade`) to score them on their own task. On current code they score worse,
because they rely on fast reverse. All D800 models share the same observation and action, so they
run unchanged on current code. Only their training reward differed.
