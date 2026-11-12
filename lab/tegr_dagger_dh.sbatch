#!/bin/bash
# Imitate downhill while the rule-based layer picks the lane, starting from la15.
# The pass side is in the subgoal, so the student can copy it from its observation.
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_dagger_dh.sbatch
#SBATCH --job-name=tegr-dagger-dh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$HOME/cbu-autonomous-robot"
export PATH="/usr/local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONPATH="src/cash_rl${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

INIT="$HOME/cbu-autonomous-robot/runs/tegr_ppo_la15_s0/final_model.zip"
test -f "$INIT"

python -m cash_rl.dagger \
  --name tegr_dagger_dh_s0 \
  --teacher downhill \
  --decision-layer \
  --init "$INIT" \
  --map cbu_floor2 \
  --pedestrians 1 \
  --encounter-prob 0.7 \
  --subgoal-lookahead 1.5 \
  --subgoal-offset 0.6 \
  --lane-hold 8 15 \
  --rounds 12 \
  --steps-per-round 60000 \
  --learning-rate 1e-4 \
  --workers 8 \
  --seed 0

RUN="$HOME/cbu-autonomous-robot/runs/tegr_dagger_dh_s0"
# 200 held-out goals. Rank by encounters with the rule, then the same rule fed by
# scan-vs-map tracks, and keep empty-floor success near the original lane follower.
for model in "$RUN/best_model.zip" "$RUN/final_model.zip" "$INIT"; do
  python -m cash_rl.evaluate "$model" \
    --map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1 \
    --decision-layer --subgoal-lookahead 1.5
  python -m cash_rl.evaluate "$model" \
    --map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1 \
    --decision-layer --scan-tracks --subgoal-lookahead 1.5
  python -m cash_rl.evaluate "$model" \
    --map cbu_floor2 --random 200 \
    --subgoal-lookahead 1.5 --subgoal-offset 0.6 --lane-hold 8 15
done
