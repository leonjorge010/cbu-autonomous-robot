#!/bin/bash
# Does holding still when the lane cannot clear a person help, on the people
# we already tested and on a mix who sometimes step aside or cross?
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_wait_eval.sbatch
#SBATCH --job-name=tegr-wait-eval
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

MODEL="$HOME/cbu-autonomous-robot/runs/tegr_dagger_dh_s0/final_model.zip"
test -f "$MODEL"
OLD=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
     --subgoal-lookahead 1.5 --decision-layer --wait)
MIX=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
     --yield-prob 0.5 --cross-prob 0.34
     --subgoal-lookahead 1.5 --decision-layer --wait)

python -m cash_rl.evaluate downhill "${OLD[@]}"
python -m cash_rl.evaluate "$MODEL" "${OLD[@]}"
python -m cash_rl.evaluate downhill "${MIX[@]}"
python -m cash_rl.evaluate "$MODEL" "${MIX[@]}"
