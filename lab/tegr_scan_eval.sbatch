#!/bin/bash
# How close the passing rule gets when it sees people in the scan instead of
# the simulator. downhill is the ceiling; la15 is the current motion policy.
# Submit from ~ : sbatch ~/cbu-autonomous-robot/lab/tegr_scan_eval.sbatch
#SBATCH --job-name=tegr-scan-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
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
COMMON=(--map cbu_floor2 --random 200 --pedestrians 1 --encounter-prob 1
        --subgoal-lookahead 1.5)

python -m cash_rl.evaluate downhill "${COMMON[@]}" --decision-layer
python -m cash_rl.evaluate downhill "${COMMON[@]}" --decision-layer --scan-tracks
python -m cash_rl.evaluate "$INIT" "${COMMON[@]}" --decision-layer
python -m cash_rl.evaluate "$INIT" "${COMMON[@]}" --decision-layer --scan-tracks
