#!/bin/bash
# Longer imitation of downhill, with the wait and a mix of people: some walk
# through the robot, some step aside, some cross the corridor at a walking pace.
# Starts from the previous dagger final. The 200-goal scores at the end are the
# old encounter test and the same mix.
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_dagger_mix.sbatch
#SBATCH --job-name=tegr-dagger-mix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$HOME/cbu-autonomous-robot"
export PATH="/usr/local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONPATH="src/cash_rl${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

INIT="$HOME/cbu-autonomous-robot/runs/tegr_dagger_dh_s0/final_model.zip"
test -f "$INIT"

python -m cash_rl.dagger \
  --name tegr_dagger_mix_s0 \
  --teacher downhill \
  --decision-layer \
  --wait \
  --init "$INIT" \
  --map cbu_floor2 \
  --pedestrians 1 \
  --encounter-prob 0.7 \
  --yield-prob 0.5 \
  --cross-prob 0.34 \
  --subgoal-lookahead 1.5 \
  --subgoal-offset 0.6 \
  --lane-hold 8 15 \
  --rounds 24 \
  --steps-per-round 60000 \
  --learning-rate 1e-4 \
  --workers 8 \
  --seed 0

RUN="$HOME/cbu-autonomous-robot/runs/tegr_dagger_mix_s0"
OLD=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
     --subgoal-lookahead 1.5 --decision-layer --wait)
MIX=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
     --yield-prob 0.5 --cross-prob 0.34
     --subgoal-lookahead 1.5 --decision-layer --wait)
for model in "$RUN/best_model.zip" "$RUN/final_model.zip" "$INIT"; do
  python -m cash_rl.evaluate "$model" "${OLD[@]}"
  python -m cash_rl.evaluate "$model" "${MIX[@]}"
done
