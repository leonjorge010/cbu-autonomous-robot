# CBU Autonomous Robot

A robot that escorts students through the CBU Engineering building using a single 2D LiDAR.
Simulation and the learned driver live here. The real robot is a Jetson Orin Nano.

`hallway` is a short practice corridor. `cbu_floor2` is the real TEGR 2nd floor. Training
happens in a fast 2D simulator on the saved map. The same policy runs in Gazebo.

The driver is two pieces. A passing rule picks a lane around a person. A small network turns
the LiDAR scan and that lane into a speed command. The current network is `tegr_dagger_dh_s0`.
Its weights are not in git.

## Jorge, Ashlyn, and scan-only

Same 200 goals on the TEGR floor, 12 Nov. Subgoal 1.5 m ahead. Ashlyn's model is
`tegr_ppo_la15_s0` and Jorge's is `tegr_dagger_dh_s0`. Both use the passing rule, with the
side held for the whole pass. Scan-only is `tegr_dagger_blind_s0`: the same kind of driver,
trained to pass from the laser and a center subgoal, and scored with the rule turned off.

![Jorge, Ashlyn, and the scan-only student on the same 200 goals](docs/jorge_ashlyn.png)

Jorge is ahead of Ashlyn on an empty corridor and crashes less when one person mixes
yielding and crossing. One person walking straight at the robot, and a crowd of four, are
ties between those two. Ashlyn finishes the successful trips about 4 seconds faster.
Scan-only reaches 99% of empty-hall goals. With one person walking straight at the robot it
reaches 52%, about 10 points above either network with the rule turned off (about 41%), and
short of Jorge's 83% with the rule on. The scripted lane follower, which reads the map,
still leads that case at 86.5%. Full times are in `WORKLOG.md`. Older models are in
`docs/MODELS.md`.

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
