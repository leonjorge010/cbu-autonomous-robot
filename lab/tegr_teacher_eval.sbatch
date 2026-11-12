#!/bin/bash
# Score the sharper teacher: a close subgoal while someone is near, a lane
# kept to the goal, and a stop that creeps aside or backs up instead of waiting
# forever. Same 200 goals as tegr-wait-eval. No new training.
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_teacher_eval.sbatch
#SBATCH --job-name=tegr-teacher-eval
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
COMMON=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
        --subgoal-lookahead 1.5 --decision-layer --wait)
MIX=(--yield-prob 0.5 --cross-prob 0.34)

python -m cash_rl.evaluate downhill "${COMMON[@]}"
python -m cash_rl.evaluate "$MODEL" "${COMMON[@]}"
python -m cash_rl.evaluate downhill "${COMMON[@]}" "${MIX[@]}"
python -m cash_rl.evaluate "$MODEL" "${COMMON[@]}" "${MIX[@]}"
