# Work log

What was done on each work day, newest first. Add a new `## YYYY-MM-DD` section at the top
for each session: what changed, results with numbers, and what's still open.

## 2026-11-12 (blind student)

Job **217** `tegr-dagger-blind` (COMPLETED, 21 min 44 s, exit 0) trained `tegr_dagger_blind_s0` on the lab. The teacher was downhill under the held-side rule. The student saw a center subgoal only (`dagger --blind`), starting from `tegr_dagger_dh_s0`, map `cbu_floor2`, 1 pedestrian, encounter 0.7, yield 0.5, cross 0.34, lookahead 1.5 m, 16 rounds × 60k steps, lr 1e-4, 8 workers, seed 0. Held-out evals used no decision layer. In-training encounter eval (100 episodes, yield and cross on) peaked at 69% and finished at 66%.

200 held-out goals, rule off. `final_model.zip` beat `best_model.zip` on every shared row.

| Test | final success / hit / timeout / time | best |
|---|---|---|
| Empty | 99.0% / 1.0% / 0% / 17.7 s | 98.5% / 1.5% / 0% / 18.0 s |
| 1 head-on | 51.5% / 48.5% / 0% / 14.7 s | 48.5% / 51.5% / 0% / 14.3 s |
| 1 mix | 71.5% / 28.5% / 0% / 16.7 s | 70.0% / 29.5% / 0.5% / 16.9 s |
| 4 head-on | 47.5% / 52.5% / 0% / 14.2 s | |
| 4 mix | 57.5% / 42.5% / 0% / 16.3 s | |

Same-job downhill with the rule reprinted job 215: 1 head-on 86.5% / 13.5% / 0% / 18.4 s, 1 mix 80% / 12% / 8% / 18.0 s. The scan-only student is about 10 points above the old rule-off net (40.5% head-on) and well short of the rule (83% / 81% / 75.5% / 65%). Empty floor held. `tegr_dagger_dh_s0` stays the default. Weights are on the lab only.

## 2026-11-12 (crashes)

Sorted the lane follower's crashes on the same 200 held-out goals (`tools/crash_causes.py`). The scores match job 215, so it is the same test. On every setting the largest pile is `lane_clear`: the lane line is a full pass away from the person, and the robot's body is still against them (edge gap about 0.05 m, person alongside).

| Test | success | collision | timeout | lane_clear share of crashes |
|---|---|---|---|---|
| 1 person, head-on | 86.5% | 13.5% | 0% | 15 / 27 |
| 1 person, mix | 80.0% | 12.0% | 8% | 18 / 24 |
| 4 people, head-on | 77.0% | 23.0% | 0% | 22 / 46 |
| 4 people, mix | 69.0% | 20.0% | 11% | 28 / 40 |

Treating that gap as blocked and yawing with no forward speed moved the head-on collisions to 16.0%. Creeping toward the passing side instead moved them to 15.5%. Neither fell, so `decision.py` is unchanged.

## 2026-11-12 (compare)

Job **216** `tegr-compare` scored `tegr_ppo_la15_s0` (Ashlyn) and `tegr_dagger_dh_s0` (Jorge) on the same 200 held-out TEGR goals. Same 1.5 m subgoal, center lane unless the held-side rule moves it, same wait. No training.

| Test | la15 success / hit / timeout / time | dagger success / hit / timeout / time |
|---|---|---|
| Empty corridor | 85.5% / 14.5% / 0% / 14.2 s | 96.5% / 3.5% / 0% / 16.5 s |
| 1 person, head-on, rule | 82.0% / 18.0% / 0% / 15.1 s | 83.0% / 16.5% / 0.5% / 19.0 s |
| 1 person, mix, rule | 76.5% / 23.5% / 0% / 15.0 s | 81.0% / 15.5% / 3.5% / 19.1 s |
| 4 people, head-on, rule | 75.5% / 24.0% / 0.5% / 15.6 s | 75.5% / 24.5% / 0% / 19.6 s |
| 4 people, mix, rule | 65.0% / 35.0% / 0% / 14.8 s | 65.0% / 32.0% / 3.0% / 18.7 s |
| 1 person, head-on, no rule | 42.0% / 58.0% / 0% / 9.5 s | 40.5% / 59.5% / 0% / 10.9 s |

With the shared rule the two models tie once people are in the corridor, except the mixed single person, where dagger crashes less (15.5% vs 23.5%). Dagger is clearly ahead on an empty corridor. la15 is about 4 s faster on every test. Neither policy avoids a head-on person without the rule (~41%).

`tools/crash_causes.py` replayed `downhill` on those same 200 goals. The success rates match job 215 on the one-person tests. Most crashes are `lane_clear`: the passing lane did not overlap the person, and the robot still hit them.

| Test | success / hit / timeout | lane_clear share of crashes |
|---|---|---|
| 1 person, head-on | 86.5% / 13.5% / 0% | 15/27 |
| 1 person, mix | 80.0% / 12.0% / 8% | 18/24 |
| 4 people, head-on | 77.0% / 23.0% / 0% | 22/46 |
| 4 people, mix | 69.0% / 20.0% / 11% | 28/40 |

## 2026-11-12

The mix-only crash jump arrived with the crossing-side update and stayed through the speed changes (jobs 211–213). Job **215** keeps the passing side for the whole pass (job 213's speeds, side chosen once). Same 200 goals. Keep: mix crashes fell 18% → 12% and timeouts stayed at 8%, under the 9% cap. Head-on did not move. No training.

| | success | collision | timeout | time |
|---|---|---|---|---|
| downhill, head-on | 86.5% | 13.5% | 0% | 18.4 s |
| dagger final, head-on | 83.0% | 16.5% | 0.5% | 19.0 s |
| downhill, mix | 80.0% | 12.0% | 8.0% | 18.0 s |
| dagger final, mix | 81.0% | 15.5% | 3.5% | 19.1 s |

Job 213's mix was 18% collisions and 2% timeouts. Job 210's mix was 11% and 9%. The lab `decision.py` is this held-side file (`94c97516cadb76979babe0b95519e404`).

`jorge/model` is pushed from the local commits (map goals and the merge). `origin/main` is still Ashlyn's `d19a6a1`. The uncommitted teacher code is not on either branch. Model training still starts from `tegr_dagger_dh_s0/final_model.zip`.

## 2026-11-11 (drive)

Job 210 left the mix timeouts at 9% and the crashes at 11%. The escape now drives a stopped person at full speed when one side has room, keeps backing up while neither side does (and holds if a wall is behind), and picks a crossing person's side again each step. Local decision tests passed (18). Job **211** finished in 7 minutes. Same four scores, job 210 in parentheses.

| | old encounters | mix |
|---|---|---|
| `downhill` | 87.0% / 13.0% hit / 0% timeout / 18.6 s (86.0 / 13.5 / 0.5 / 18.5) | 80.5% / **17.5%** hit / **2.0%** timeout / 18.1 s (80.0 / 11.0 / 9.0 / 18.0) |
| dagger final | 83.0% / 17.0% / 0% / 19.0 s (83.5 / 16.0 / 0.5 / 19.4) | 80.0% / 19.5% / 0.5% / 19.1 s (81.0 / 16.5 / 2.5 / 19.2) |

Full speed cut the mix timeouts and turned them into crashes. Job **212** turns clear first, then drives. It did not help: mix `downhill` 80.0% / 18.0% hit / 2.0% timeout / 18.0 s, and the dagger final 80.0% / 20.0% / 0% / 19.0 s. Head-on stayed 86.5% / 13.5%. Job **213** passes a stopped person at 60% speed until they are 0.8 m to the side, and only then goes full speed. It finished identical to 212: mix `downhill` 80.0% / 18.0% hit / 2.0% timeout, dagger final 80.0% / 20.0% / 0%. Head-on stayed 86.5% / 13.5%. The speed tweaks are not the lever. The mix-only crash jump arrived with the crossing-side update and did not move after that. Stop. Job 210 (creep, side held) is the better teacher of the night: mix crashes 11% rather than 18%, timeouts 9%. The lab file is the job 213 rule. Do not train.

## 2026-11-11 (teacher)

The stop is not the missing piece (job 207), and another imitation of it gained about a point (job 208). Keep `tegr_dagger_dh_s0/final_model.zip`. The teacher now holds still for 0.3 s, keeps holding if the person is walking off the lane, otherwise creeps to the side that has room, and backs up for at most 2 s when the corridor has no such side. While someone is closing inside 3 m the subgoal and the followed lane pull in to 0.5 m, and a lane stays on through the last metres while someone is still ahead. Local tests passed (96). Lab copies of `decision.py` and the raycast hook compiled. Job **210** `tegr-teacher-eval` finished in 7 minutes (`COMPLETED`, exit 0). Same 200 goals as 207, decision layer and the new wait on. Job 207 in parentheses.

| | old encounters | mix (yield 0.5, cross 0.34) |
|---|---|---|
| `downhill` | 86.0% / 13.5% hit / 0.5% timeout / 18.5 s (was 84.0 / 16.0 / 0 / 17.8) | 80.0% / 11.0% hit / 9.0% timeout / 18.0 s (was 81.0 / 11.0 / 8.0 / 18.0) |
| dagger final, same weights | 83.5% / 16.0% hit / 0.5% timeout / 19.4 s (was 75.5 / 24.0 / 0.5 / 18.0) | 81.0% / 16.5% hit / 2.5% timeout / 19.2 s (was 76.0 / 23.5 / 0.5 / 18.3) |

`downhill` did not beat job 207: head-on crashes fell 2.5 points and the mix was unchanged, timeouts included. The learned weights, with no new training, gained about 8 and 5 points because the nearer subgoal and the creep/backup now steer them. About a second slower. Do not start another imitation until the teacher itself beats 207.

## 2026-11-11 (later)

The 17.5% collisions left on `downhill` are the passing rule with no way to stop. Wait is a wrapper, so it can be scored on the existing dagger final without retraining: if the chosen lane still leaves a person inside the robot, the command is held at zero until they have passed or a lane opens (`DecisionLayer(wait=True)`, `evaluate --wait`). People are no longer one behavior. Each person is the old non-yielding walker, a yielder (`--yield-prob`), or a crosser at 1.0–1.4 m/s (`--cross-prob`); both probabilities default to 0 so old seeds stay the same.

Lab jobs, not committed. **207** `tegr-wait-eval` finished. Same 200 held-out goals, decision layer, wait on. Previous numbers without the wait were downhill 82.5% / 17.5% hit / 17.5 s and the dagger final 79.0% / 21.0% / 18.2 s.

| | old encounters + wait | mix + wait (yield 0.5, cross 0.34) |
|---|---|---|
| `downhill` | 84.0% / 16.0% hit / 0% timeout / 17.8 s | 81.0% / 11.0% hit / **8.0% timeout** / 18.0 s |
| dagger final | 75.5% / 24.0% hit / 0.5% timeout / 18.0 s | 76.0% / 23.5% hit / 0.5% timeout / 18.3 s |

The stop is a small gain for `downhill` on the old test (time barely moves) and a small loss for the student, inside the usual ±3% run variation. On the mix, `downhill` trades crashes for timeouts: people who also stop leave both sides waiting. **208** `tegr-dagger-mix` finished in about 40 minutes of training (24 rounds × 60k, lr 1e-4, init the dagger final, teacher `downhill`, wait on, the mix). In-loop 100-episode mix check sat at 81–84% (best round 84%). The 200-goal ranking, stop on:

| | old encounters | mix |
|---|---|---|
| `tegr_dagger_mix_s0` final | 76.5% / 22.5% hit / 1.0% timeout / 17.3 s | 77.5% / 20.0% hit / 2.5% timeout / 17.5 s |
| `tegr_dagger_mix_s0` best | 76.5% / 22.5% / 1.0% / 17.4 s | 77.0% / 20.5% / 2.5% / 17.5 s |
| previous final (same run) | 75.5% / 24.0% / 0.5% / 18.0 s | 76.0% / 23.5% / 0.5% / 18.3 s |

About a point better and slightly faster, inside the ±3% run variation. Keep the previous final. The next gain has to come from a better teacher.

## 2026-11-11

Gentle PPO fine-tunes of `tegr_ppo_la15_s0` did not move the actor. Job 187 (`tegr-ft1e4`, lr 1e-4, 750k) and job 188 (`tegr-ft5e5`, lr 5e-5, 1.5M) stayed on the frozen starting weights through the 200k value warmup (95% / 5% collisions / +54.3 on the 40-goal in-training eval) and after unfreeze sat at 88–92% success with 5–10% collisions. Their `best_model.zip` files are the original la15 policy. Stopped fine-tuning.

Two gaps, handled separately. Motion: DAgger imitates `downhill` while the rule-based decision layer sets the lane, initialized from la15 (`--teacher downhill --decision-layer --init`). Decision: `cash_rl.tracks.ScanDecisionLayer` runs that same passing rule on tracks from a 1° scan compared with the map (`raycast(include_pedestrians=False)`), not on `env.pedestrians`. `evaluate --scan-tracks` selects it. Unit tests in `test/test_tracks.py`.

Lab jobs (not committed): **197** `tegr-scan-eval` died on `decision.py` (`IndexError` when a pedestrian index outlived the list). The lane memory now drops indexes past the end of the list (`test_a_stale_lane_index_is_dropped_when_a_pedestrian_disappears`). **198** `tegr-dagger-dh` finished on `cbu-5090-right` (best in-loop round 87%). **200** `tegr-scan-eval` finished on `cbu-5090-left`. 200 held-out encounter goals, then empty floor:

| | privileged | scan tracks | empty floor |
|---|---|---|---|
| `downhill` | 82.5% / 17.5% hit | 80.5% / 19.5% | — |
| `tegr_dagger_dh_s0` final | **79.0%** / 21.0% | **78.5%** / 21.5% | **100%** / 0% |
| `tegr_dagger_dh_s0` best | 76.5% / 23.5% | 76.0% / 24.0% | 99.0% / 1.0% |
| la15 s0 | 67.5% / 32.5% | 65.0% / 35.0% | 98.0% / 2.0% |

`final_model.zip` clears the keep bar (encounters above ~72.5%, empty floor near 96%). It closes most of the gap from la15 to `downhill`. Scan tracks cost `downhill` 2 points and the student half a point, so the 1° tracker is enough for this rule. Not committed.

## 2026-11-02

Covers 11-01 (from 14:43) through 07:04 on 11-02: two queues, `queue_lane_retrain.txt` and
`queue_9h.txt`, all jobs OK.

**Lane retrain works: `tegr_ppo_la15`** (subgoal 1.5 m ahead instead of 3 m, random lanes held
8–15 s instead of 2–6 s, lane penalty 2.0; otherwise as `tegr_ppo_lane_w2_s0`). 200 held-out
TEGR goals, 3 seeds (s0 / s1 / s2):

| | s0 | s1 | s2 | mean | w2 (3 m) | `downhill` (1.5 m) |
|---|---|---|---|---|---|---|
| No pedestrians, random lanes | 96.0% | 97.0% | 98.5% | **97.2%** | 93.5% | — |
| `lane` error, lane +0.5 / 0 / −0.5 (m) | .22/.17/.19 | .24/.18/.25 | .21/.15/.20 | **.22/.17/.21** | .49/.25/.26 | .15/.12/.14 |
| Encounter, center lane | 49.0% | 52.5% | 50.5% | **50.7%** | 50.5% | 53% |
| Encounter, rule-based layer | 72.5% | 64.0% | 70.5% | **69.0%** | 53.0% | 91% |
| Encounter, learned subgoal agent | 65.5% | — | — | — | 53.0% | 73.5% |

PPO now follows lanes (error close to `downhill`), so the upper half of the hierarchy finally
helps: the rule-based layer adds +18 points (was +2.5). Learned subgoal agent over la15 s0
(`tegr_subgoal_la15_s0`): 65.5% on encounters, 95.5% without pedestrians but return +33 vs +60
and lane error 0.43 — it switches lanes with nobody in the way.

**Right-lane collisions** (s0, lane −0.5: 9.5%): 17 of 19 hit the wall on the robot's right,
clustered where walls jut out (4 at the same spot, ~(16, 12.8)), mostly where the lane penalty is
waived (turns, junctions, last metres). Largely seed-specific: s1 / s2 right lane 4% / 1% (s1
left lane 8.5%). Tools: `view_env --lane-offset X --held-out I...` replays evaluate's held-out
episode `random_I` (`cae8cf3`); `src/cash_rl/tools/lane_collisions.py` lists which episodes
collide (uncommitted).

**What still fails is the pedestrian:** la15 s0 + rule-based layer, 55 encounter failures =
51 pedestrian hits + 4 walls. la15 never trained with pedestrians.

**Pedestrian fine-tune: `tegr_ppo_la15_ped_s0`** (new `train --init ZIP` and
`--decision-layer-prob P`, `5e68926`; `DecisionLayer(prob=P)` runs the rule-based layer in that
share of episodes, random lanes in the rest). From la15 s0, 3M steps, 1 pedestrian, encounter
0.7, rule-based layer in 70% of episodes, la15's own hyperparameters (lr 3e-4). It got worse:

| | la15 s0 | fine-tune final | checkpoints 250k / 500k / 1M |
|---|---|---|---|
| Encounter, rule-based layer | 72.5% | 66.5% | **77.5%** / 76.5% / 72.5% |
| + safety layer | — | 68.0% | |
| No pedestrians, random lanes | 96.0% | 92.5% | |
| Lane −0.5 pinned | 90.5% | 86.0% | |
| Learned subgoal agent on top (`tegr_subgoal_la15_ped_s0`), encounter | 65.5% | 59.0% | |

Periodic eval fell steadily from 92.5% at step 0 to ~80%. Pedestrian training helps briefly
(+5 by 250k) and then the full learning rate erodes what la15 knew — likely too large a step for
fine-tuning, with the value head facing reward terms (pedestrian penalties) it never saw.
Best pedestrian motion agent for now: `runs/tegr_ppo_la15_ped_s0/checkpoints/model_249996_steps.zip`.

**Next steps**
1. Redo the fine-tune gently: lr ~5e-5–1e-4 (new `train --learning-rate LR`, which also
   overrides an `--init` model's rate), ~750k steps, checkpoints every 100–125k. Queued as
   `runs/queue_gentle_ft.txt` (lr 1e-4 for 750k and 5e-5 for 1.5M, from la15 s0/s1/s2, every
   checkpoint on the encounter test + rule-based layer and without pedestrians; plus the
   unseen-hallway test from step 5). Rank checkpoints by the
   encounter test with the rule-based layer — the periodic eval uses random pedestrian starts.
   Optionally warm up the value head first (a few 100k steps with the policy frozen).
2. If it beats ~77.5% / holds ~96% without pedestrians, repeat on la15 s1 and s2 for a 3-seed mean.
3. Retrain the learned subgoal agent over the best motion agent, with a cost for changing lanes
   when no pedestrian is near (it wanders: lane error 0.43 with nobody around).
4. More pedestrians: keep training at 1 until encounters with a decision layer reach ~85%, then
   fine-tune 1 → 2 → 4. Meanwhile add 4-pedestrian evals (rule-based layer) of the best model as a
   baseline — not added to `queue_9h.txt` by choice.
5. Unseen-environment test (proposal week 9): evaluate the TEGR models on the hallway map
   (`--map hallway --random 200`), eval only.
6. Optional: the right-lane wall clips — keep the lane `SAFE_CLEARANCE` from walls that jut out
   ahead, not only at the cross-section at the robot.
7. Update `docs/MODELS.md` and the README with la15 (3 seeds) and the fine-tune.

## 2026-10-31

Covers the evening of 10-30 through 01:51 on 10-31.

**Learned subgoal agent** (hierarchy upper half, RL; `cash_rl/subgoal_agent.py`, console script
`train_subgoal`, `d8ba369`). Ashlyn's choices: learned with RL, action = lane offset + wait,
inputs = LiDAR only. Built in a worktree and merged after `queue_later.txt` finished.
- Every 5 control steps (0.5 s) it picks a lane (±1 m, clipped by the env) and whether to wait.
  Observes the 36 beams now / 0.5 s / 1 s ago, the planner's 3 m center-line subgoal, the end
  goal, lane room left/right, current lane, velocity, previous action — no pedestrian positions.
  Reward: task reward over its steps minus the motion agent's lane penalty.
- `SubgoalEnv` trains it over a frozen motion policy (model zip via a numpy actor, or `downhill`);
  `SubgoalLayer` runs it in `evaluate`/`view_env --subgoal-agent ZIP`. 7 tests.

**`queue_later.txt` results** (all OK, 17:57–19:47): `tegr_ppo_lane_w2_s0` (lane penalty 2.0).

| 200 goals, TEGR | w2 (2.0) | lane_s0 (0.5) | `downhill` |
|---|---|---|---|
| No pedestrians, random lanes | 93.5% | 94.0% | — |
| Lane +0.5 pinned: success / `lane` error | 93.5% / 0.49 m | 93.0% / 0.41 m | 100% / 0.16 m |
| Center: success / error | 91.5% / 0.25 m | 93.0% / 0.23 m | 100% / 0.13 m |
| Lane −0.5 pinned: success / error | 90.0% / 0.26 m | 90.5% / 0.37 m | 100% / 0.15 m |
| Encounter, center lane | 50.5% | 47.5% | 53% |
| Encounter, decision layer | 53.0% | 51.0% | 91% |

Raising the penalty kept goal-reaching but did not fix lane following.

**Why PPO won't follow lanes** (`src/cash_rl/tools/lane_diagnostics.py`: `pinned`, `sensitivity`,
`tug`, `rewards`, `pursuit`):
- Not the penalty's size: on the +0.5 lane w2 pays ~32 lane penalty per episode vs ~7 for
  `downhill`, and still drives at +0.09 m from center (+0.5 / 0 / −0.5 lane → +0.09 / −0.07 /
  −0.22 m, ~30% of the shift).
- Not training length: checkpoints 500k–3M all cover 0.2–0.3 m of the 1.0 m shift; plateaued by
  500k steps.
- Not the reward: `downhill` earns the same non-lane reward following lanes or ignoring them
  (76.5 vs 76.8 per episode), so following is worth +21 with no hidden cost.
- The policy does react to the subgoal (0.27 rad/s per 9.5° of bearing, vs 0.07 for `tegr_ppo`
  without lanes), but steers about twice as hard from its own LiDAR position: placed on the
  ±0.5 lane it steers back toward center (±0.42 rad/s). The penalty is charged on only ~30% of
  steps (waived in turns, junctions, open areas, last 3 m), so it mostly learned centering.
- The cue is weak: steering straight at the observed 3 m subgoal follows no better than PPO
  (|err| 0.36 vs 0.32 m). 0.5 m sideways is only ~9° at 3 m, closing the gap takes ~6 s of
  driving, and lanes redraw every 2–6 s. Same pursuit controller: 1.5 m lookahead with 8–15 s
  holds → **0.20 m** (= `downhill`); 3 m / 2–6 s 0.36, 3 m / 8–15 s 0.30, 1.5 m / 2–6 s 0.25.
- No penalty for switching lanes as such; a 0.5 m switch costs ~5–10 while the robot moves over.

**Lane hold flag** (`7d92401`, worktree, merged at 01:51): `--lane-hold MIN MAX` on
`train`/`dagger`/`evaluate`/`view_env` (default 2–6 s, so old runs reproduce; saved in
`config.json`, `evaluate`/`view_env` default to the model's). `train_subgoal` now runs the motion
model at its own lookahead (`motion_lookahead`; `downhill` 3 m) instead of a fixed 3 m. 85 tests pass.

**`queue_subgoal_agent.txt` results** (all OK, 22:13–01:51; each agent 1M decisions, ~105 min).
TEGR encounter test, 1 pedestrian, 200 held-out goals:

| Motion agent | Center lane | Rule-based layer | Learned subgoal agent |
|---|---|---|---|
| `tegr_ppo_lane_w2_s0` | 50.5% | 53% | **53.0%** (`tegr_subgoal_w2_s0`) |
| `downhill` | 53% | 91% | **73.5%** (`tegr_subgoal_dh_s0`) |

- Over `downhill` the learned agent adds 20 points but trails the rule-based layer (privileged
  pedestrian positions) by 17.5; its periodic eval was flat at ~70% at the end.
- Over w2 it can't beat the follower's weakness.
- No pedestrians, w2 + agent: 94.5% (no cost to goal-reaching).

**Queue** `runs/queue_lane_retrain.txt` (~4–5 h, not started): `tegr_ppo_la15_s0` (1.5 m
lookahead, 8–15 s holds, otherwise as w2), lane L/C/R vs `downhill`, encounter center vs decision
layer, then `tegr_subgoal_la15_s0` over it with encounter and no-pedestrian evals.

**Open**
- `queue_lane_retrain.txt`: does `tegr_ppo_la15_s0` follow lanes (`lane` toward ~0.2 m) and keep
  ~93% goal-reaching with less corner warning at 1.5 m? Then seeds 1–2.
- Learned subgoal agent gap to the rule-based layer (73.5 vs 91% over `downhill`): longer
  training, or a larger LiDAR history window.
- `docs/MODELS.md` not updated for any lane / subgoal-agent model; `main` is 11 commits ahead of
  `origin`, not pushed.

## 2026-10-30 (evening)

**Rule-based decision layer** (hierarchy upper half, Ashlyn; `cash_rl/decision.py`, `49aeb89`).
Built in a separate worktree (`decision-layer` branch) so the running queue kept `main`'s code;
fast-forwarded into `main` and rebuilt after the queue finished (`b324cff`).
- `PedestrianLaneDecider`: center lane until a pedestrian is ahead (≤6 m, within 1.5 m of the
  robot's line) and closing (oncoming or overtaken). Then the lane goes 0.8 m beside the
  pedestrian's line on the side with more room where they will meet (±1 m; keep right on ties),
  held until the pedestrian is 0.8 m behind. `DecisionLayer` wrapper sets `env.lane_offset` each
  step and rebuilds the obs; `evaluate`/`view_env --decision-layer`. Privileged (true pedestrian
  state); on the robot it needs LiDAR tracking. 6 tests (`test/test_decision.py`).
- Three fixes took `downhill` + layer from 77% to 92% (100 eps): judge room at the predicted
  meeting point, not beside the pedestrian now (the hallway fountain narrows the right side);
  place the lane relative to the pedestrian, not a fixed ±0.6 m from center; measure from the
  followed lane's center (the center abreast of the pedestrian jumps at door recesses/open areas).
- Limits: no "wait" (dead-center pedestrian in the 2.4 m hallway or by the fountain leaves no lane
  wide enough), no lanes in the last 3 m (5 of the last 18 hits), ignores pedestrians from behind.

TEGR encounter test, 1 pedestrian, 200 held-out goals:

| Motion agent | Center lane | Decision layer |
|---|---|---|
| `downhill` (pure lane follower, privileged field) | 53.0% | **91.0%** |
| avoider (privileged) | 89.0% (10-23) | 94.0% |
| `tegr_ppo_lane_s0` (lane penalty 0.5) | 47.5% | 51.0% |

The layer works when the motion agent follows the lane; `tegr_ppo_lane_s0` barely does (lane error
0.41 m with the layer).

**`tegr_ppo_lane_s0`** (TEGR PPO, 3M steps, 3 m subgoals, random lanes ±0.6 m, lane penalty 0.5;
165 min incl. ~1 h with the laptop lid closed — closing the lid pauses runs, doesn't break them).
- Goal-reaching, no pedestrians, random lanes: **94.0%** final / 95.5% best, collisions 4.5%,
  reversing 6.2%. `tegr_ppo` (no subgoals) 3 seeds: 84.3% final, collisions 9.5–14%. One seed.
- Lane following is weak: pinned lane +0.5 / 0 / −0.5 → robot drives at +0.12 / −0.03 / −0.19 m
  from center (median, 40 eps), ~1/3 of the shift. `lane` error 0.37–0.41 m vs `downhill`
  0.13–0.16 m (`runs/eval_tegr_ppo_lane_s0_{left,center,right}.txt`).
- So `SUBGOAL_LANE_WEIGHT` 0.5 → **2.0** (`b324cff`). Scripted check, TEGR random lanes: follows 93
  vs ignores 59 (was 108 vs 94); stops −189. All tests pass. Old runs keep 0.5 in `config.json`;
  re-evaluating them with 2.0 changes only the `return` column.

**`queue_now.txt` results** (all 12 jobs OK, 13:03–17:41; TEGR, 200 held-out goals, 3 seeds)
- Experiment A control, hallway-trained → TEGR zero-shot: `ppo_nosg` (same code, no subgoals)
  56.8% (57.0/54.5/59.0), timeouts 37.8%, vs `ppo_sg3` **72.5%**, timeouts 16.5%. Subgoals +15.7
  points on the unseen floor; the 10-30 confound (old 41-dim `ppo_d800_prox`) is gone.
- Safety layer on `tegr_dagger_sg3`: no pedestrians 88.7% → **94.7%** (collisions 10–15% →
  0.5–1.5%, timeouts 4–5%); encounter 68.8% → **73.3%**; overrides 6–10% of steps.
- 4-pedestrian encounter test: `tegr_dagger_sg3` (trained with 1) 63.7% (62.5/61.0/67.5),
  `downhill` 47.0%, avoider 87.5%.

**Queue** `runs/queue_later.txt` (6 jobs, ~2 h, not started): `tegr_ppo_lane_w2_s0` (penalty 2.0),
lane pinned left/center/right vs `tegr_ppo_lane_s0` and `downhill`, encounter test center lane vs
`--decision-layer`. Seeds 1–2 only if it follows.

**Open**
- `queue_later.txt`: does `tegr_ppo_lane_w2_s0` follow lanes (`lane` toward ~0.15 m) while keeping
  ~94% goal-reaching, and does the decision layer lift its encounter score well above center lane?
- Decision layer: a "wait" option, lanes within the last 3 m, pedestrians from behind, LiDAR
  pedestrian tracking for the robot.
- `evaluate` metrics for side-keeping and off-lane distance near pedestrians.
- Add the 10-30 models to `docs/MODELS.md`. Push `main` (8 commits ahead of origin).

## 2026-10-30 (afternoon)

**Subgoal lanes** (hierarchy step 3, Ashlyn; commit `134f9fb`). The motion agent should follow
subgoals from the decision layer (e.g. steer right around a pedestrian even when the goal is
ahead-left) and see the end goal only once it is within the 3 m lookahead.
- `--subgoal-offset W` on `train`/`dagger`/`evaluate`/`view_env`: the 3 m subgoal sits in a lane up
  to W m left/right of the corridor center. The center comes from the walls, so the lane doesn't
  drift with the robot. The offset is cut back to keep 0.6 m (`SAFE_CLEARANCE`) from obstacles, so
  the max is ~0.6 m in a 2.4 m corridor. It is redrawn every 2–6 s (30% center).
  `reset(options={'lane_offset': x})` / `evaluate --lane-offset x` pin it; a decision layer sets
  `env.lane_offset`.
- Reward: `spec.subgoal_lane_penalty`, 0.5 per m beyond 0.1 m from the lane at the robot's own
  position, per step. Waived in turns (>30°), junctions, open areas and the last 3 m. Measuring
  1 m ahead instead penalized a robot correctly holding the outside of a left turn by ~18 return.
- Following a lane away from the goal isn't penalized: progress telescopes, so a detour costs
  only its extra time. Scripted lane follower, hallway, right vs left lane: door ahead-left 106.9
  vs 103.2; left turn up the branch 125.1 vs 121.7; cutting across a right lane 53.7 / 70.7.
  Pinned by `test_following_a_lane_away_from_the_goal_is_not_penalized`. TEGR, random lanes,
  30 goals: follows 108, ignores 94, stops −121. Runs without lanes: reward unchanged.
- `downhill`/`avoider` follow the lane, so DAgger could teach it too. Within 2 m of a pedestrian
  the avoider may leave the lane, but an off-center lane sets its passing side (replaces
  keep-right). Avoider passes on the lane's side 86–91% on TEGR (hallway 62–79%), 95% success with
  random lanes (93% without).
- Env with subgoals + lanes: 359 steps/s vs 593 without (TEGR) → ~2.2 h per 3M-step PPO run.
- 6 new tests; 71 pass.

**Decisions**
- Motion agent trained with PPO and rewarded for following subgoals, not DAgger: PPO never beat
  `downhill` on encounters, DAgger plateaus at 68.8% (teacher 89%), and the course needs an RL
  stage. Phase 1: no pedestrians, random lanes so paths aren't straight. Phase 2: test with
  pedestrians and a (rule-based) decision layer that moves the lane around them. Hybrid kept: the
  motion agent may still avoid on its own when the lane is wrong, but takes its side from the lane.
- Geometry limit: a 0.6 m lane doesn't clear a pedestrian walking down the middle of a 2.4 m
  corridor (~0.35 m slack). The decision layer needs bigger offsets or a "wait" subgoal.

**Queues** (`runs/queue_now.txt` started 13:03, ~3.8 h): `tegr_ppo_lane_s0` (TEGR, 3M steps,
`--subgoal-lookahead 3 --subgoal-offset 0.6`) + lane pinned left/center/right evals
(`runs/eval_tegr_ppo_lane_s0_{left,center,right}.txt`, `downhill` as reference); experiment A
control `ppo_nosg_s0-2` → `runs/eval_tegr_zeroshot_nosg.txt` (vs `ppo_sg3` 72.5%); `--safety` on
`tegr_dagger_sg3` → `runs/eval_tegr_{enc,p0}_dagger_sg3_safety.txt`; 4-pedestrian encounter
baselines → `runs/eval_tegr_enc4_{dagger_sg3,scripted}.txt`. `runs/queue_later.txt` (~4.5 h):
lane seeds 1–2 and the 3-seed lane evals. Removed: 4-pedestrian DAgger without lanes, the stale
entropy-0.01 runs in `runs/queue.txt`.

**Open**
- Results of `queue_now.txt`: does `tegr_ppo_lane_s0` reach goals (vs `tegr_ppo` 84.3%) and follow
  lanes (`lane` column vs `downhill`)? Then `queue_later.txt`.
- Rule-based decision layer (lane away from an oncoming pedestrian, back to center after), then
  the phase-2 pedestrian test. On the robot it needs pedestrian detection from LiDAR.
- Within 3 m of the goal the lane is off, so the decision layer can't steer there.
- `evaluate` metrics for side-keeping and off-lane distance near pedestrians.
- Add the 10-30 models to `docs/MODELS.md`. Push `main` (5 commits ahead of origin).

## 2026-10-29

**Merged `pedestrians-no-yield` into `main`** (Ashlyn). Committed the map-generic TEGR code
(`14b094d`: `--map`, keepout zones, per-map pedestrian axes, limited Dijkstra), then merged the
branch. `main` now has non-yielding pedestrians, the lane penalty, `--encounter-prob`, the
privileged `avoider`, `dagger` and TEGR training. 63 tests pass. No new training or results.
Pushed `main` to GitHub (`95f607b..8cee225`); deleted the merged local branches
`pedestrians-no-yield` (never pushed) and `pedestrians-scan-stack`.

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

**Decisions**
- Hierarchy: the decision (subgoal) layer is the geodesic planner (`route()`), not a learned model.
  It uses only the map and the robot's pose, which the real robot will have (SLAM map + Nav2
  localization). Report it as a deliberate change from the proposal's learned subgoal agent.
  The planner ignores pedestrians; a decision layer that picks a side (or waits) for an oncoming
  pedestrian is the extension if experiment B shows the motion model still can't commit to a
  side. Rule-based first, learned if time allows; a fully learned subgoal agent is future work.
- Warm starts vs. training from scratch: keep scratch runs for every comparison in the report
  (equal footing, ±3% seed noise). Warm-start only to build the final model, as a recorded
  curriculum: DAgger student → PPO fine-tune (1 pedestrian) → 4 pedestrians. Train from scratch
  after reward/obs changes (stale value function, old habits: `ppo_peds` fell 90.5% → 36% when
  pedestrians stopped yielding).

**Overnight queue** (`runs/queue_subgoal.txt`, all 14 jobs OK, 00:36–08:26; hallway PPO with
subgoals took 40–44 min per 1.5M steps (route tracing; `ppo_d800_prox` took 11), DAgger 20 rounds
48–63 min)
- A, unseen map: `ppo_sg3_s0/1/2`, hallway PPO with `--subgoal-lookahead 3` (`ppo_d800_prox`
  settings, 1.5M steps), then zero-shot on TEGR → `runs/eval_tegr_zeroshot_sg3.txt`. Compare with
  `ppo_d800_prox`'s 60% (34% timeouts).
- B, pedestrians on TEGR: DAgger 20 rounds, 1 pedestrian, seeds 0–2, with 3 m subgoals
  (`tegr_dagger_sg3_s*`) and without (`tegr_dagger_r20_s*`, also the seed/plateau check of 61.5%).
  Encounter test and no-pedestrian test → `runs/eval_tegr_{enc,p0}_dagger_{sg3,r20}.txt`.

**Results** (200 held-out goals per model, 3 seeds each, final models)

B, TEGR DAgger (clean comparison: identical except subgoals):

| Test | `tegr_dagger_sg3` (3 m) | `tegr_dagger_r20` (none) | `downhill` | avoider |
|---|---|---|---|---|
| Encounter, 1 ped | **68.8%** (66.5/69.0/71.0) | 64.8% (63.0/65.0/66.5) | 50.5% | 89.0% |
| No pedestrians | 88.7% (88.5/88.0/89.5) | 87.2% (84.5/88.0/89.0) | 100% | — |

- Subgoals +4 points on the encounter test; every seed beats its no-subgoal partner (+2.5 to +4.5).
  No difference without pedestrians; failures there are all wall collisions (10.5–15.5%).
- The 10-23 `tegr_dagger_p1` (61.5%) holds up: 3-seed mean at 20 rounds 64.8%.
- All 6 DAgger models beat `downhill` by 14–20 points.
- DAgger has plateaued: per-round eval (100 eps) rises to ~round 10, then wanders 57–70% with no
  trend; MSE ~0.10. More rounds won't close the gap to the teacher.

A, unseen map (hallway-trained, TEGR zero-shot):

| Model | Hallway | TEGR zero-shot | TEGR timeouts |
|---|---|---|---|
| `ppo_sg3` (3 m, 3 seeds) | 93.0% (95.5/88.0/95.5) | **72.5%** (73.0/65.0/79.5) | 16.5% |
| `ppo_d800_prox` (1 seed, 10-15 code) | 96.5% | 60.0% | 34.0% |

- Subgoal models generalize better (+12.5 points, timeouts halved) with no loss on the hallway.
- **Confounded:** `ppo_d800_prox` predates scan stacking (41-dim obs, 1 scan) and its seeds 1/2 are
  only in the GitHub release. Needs a same-code control without subgoals before claiming it.

**Open**
- Control for A (3 seeds, ~40 min each; not started). Write `runs/queue_control.txt` with
  `runs/train_eval.sh ppo_nosg_s{0,1,2} --goal-distance-range 1 25 --junction-prob 0.3 --timesteps
  1500000 --seed {0,1,2}` (one line per seed) plus
  `OMP_NUM_THREADS=1 ros2 run cash_rl evaluate runs/ppo_nosg_s{0,1,2}/final_model.zip --map cbu_floor2
  --random 200 > runs/eval_tegr_zeroshot_nosg.txt 2>&1`, then
  `tmux new -d -s queue "QUEUE=runs/queue_control.txt ~/ros2_ws/runs/queue.sh"`. Compare with 72.5%.
- Add these models to `docs/MODELS.md`.
- Gap to the teacher (68.8% vs 89%) won't close with more DAgger rounds: discrete 54-action
  student, and/or a side-choosing decision layer; check the `sg3` students' hard-swerve rate.
- `--init-from` (PPO fine-tune from `tegr_dagger_sg3`) for the final model.
- TEGR wall collisions: `evaluate --safety` on the TEGR models; longer training.
- Why hallway models time out on TEGR (34%; reverse ~18% of steps) — not the straight-line goal.
- Then 4 pedestrians. Carried over: keep-right teacher at 4 peds (78%), `demo` hallway doors,
  stale `runs/queue.txt`, `elevator_2` goal position (check on site).

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
