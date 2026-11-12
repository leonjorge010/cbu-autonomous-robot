#!/bin/bash
# Same 200 held-out goals for Ashlyn la15 and Jorge dagger_dh.
# Empty corridor, one person head-on, the mixed crowd, then four people.
# The decision layer and the wait are on for every pedestrian test except the
# last pair, which is the policy alone. No new training.
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_compare.sbatch
#SBATCH --job-name=tegr-compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$HOME/cbu-autonomous-robot"
export PATH="/usr/local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONPATH="src/cash_rl${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

ASH="$HOME/cbu-autonomous-robot/runs/tegr_ppo_la15_s0/final_model.zip"
JORGE="$HOME/cbu-autonomous-robot/runs/tegr_dagger_dh_s0/final_model.zip"
test -f "$ASH"
test -f "$JORGE"

BASE=(--map cbu_floor2 --random 200 --subgoal-lookahead 1.5 --subgoal-offset 0)
HEAD=(--pedestrians 1 --encounter-prob 1 --decision-layer --wait)
MIX=(--yield-prob 0.5 --cross-prob 0.34)
CROWD=(--pedestrians 4 --encounter-prob 1 --decision-layer --wait)
RAW=(--pedestrians 1 --encounter-prob 1)

for MODEL in "$ASH" "$JORGE"; do
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}"
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}" "${HEAD[@]}"
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}" "${HEAD[@]}" "${MIX[@]}"
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}" "${CROWD[@]}"
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}" "${CROWD[@]}" "${MIX[@]}"
    python -m cash_rl.evaluate "$MODEL" "${BASE[@]}" "${RAW[@]}"
done
