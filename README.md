# CBU Autonomous Robot

A robot that escorts students through the CBU Engineering building using a single 2D LiDAR.
Simulation and the learned driver live here. The real robot is a Jetson Orin Nano.

`hallway` is a short practice corridor. `cbu_floor2` is the real TEGR 2nd floor. Training
happens in a fast 2D simulator on the saved map. The same policy runs in Gazebo.

The driver is two pieces. A passing rule picks a lane around a person. A small network turns
the LiDAR scan and that lane into a speed command. The current network is `tegr_dagger_dh_s0`.
Its weights are not in git.

## Result

200 held-out goals on the TEGR floor, 12 Nov. The subgoal is 1.5 m ahead. The passing side
is held for the whole pass. Numbers are goals reached.

| | Empty | One person, head-on | One person, mixed | Four people, head-on | Four people, mixed |
|---|---|---|---|---|---|
| `tegr_ppo_la15_s0` | 85.5% | 82% | 76.5% | 75.5% | 65% |
| `tegr_dagger_dh_s0` | **96.5%** | **83%** | **81%** | 75.5% | 65% |
| Scripted lane follower | — | **86.5%** | 80% | 77% | 69% |

The empty-corridor column keeps the lane in the center. With the rule turned off, both
learned models reach the goal about 41% of the time. The rule does the passing. Collision
and time numbers are in `WORKLOG.md`. Older models are in `docs/MODELS.md`.

## Where things are

| Path | What it is |
|---|---|
| `src/cash_sim/worlds/cbu_floor2.sdf` | TEGR floor in Gazebo |
| `src/cash_sim/worlds/hallway.sdf` | Practice hallway |
| `maps/cbu_floor2.*` | Map the 2D trainer and the passing rule use |
| `src/cash_rl/cash_rl/spec.py` | Observation, action, reward, collision |
| `src/cash_rl/cash_rl/raycast_env.py` | Fast 2D simulator |
| `src/cash_rl/cash_rl/decision.py` | Passing rule |
| `src/cash_rl/cash_rl/train.py` | Training |
| `WORKLOG.md` | Session notes, newest first |
