#!/bin/bash
# Gentle pedestrian fine-tune of tegr_ppo_la15 seed 0 at lr 5e-5.
# Queued behind job 186: it needs that run's final_model.zip.
# Submit from ~ so the log lands in ~: sbatch ~/cbu-autonomous-robot/lab/tegr_ft_5e5.sbatch
#SBATCH --job-name=tegr-ft5e5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --dependency=afterok:186

set -euo pipefail
cd "$HOME/cbu-autonomous-robot"
export PATH="/usr/local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONPATH="src/cash_rl${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m cash_rl.train \
  --init "$HOME/cbu-autonomous-robot/runs/tegr_ppo_la15_s0/final_model.zip" \
  --name tegr_ppo_la15_ft5e5_s0 \
  --map cbu_floor2 \
  --learning-rate 5e-5 \
  --timesteps 1500000 \
  --checkpoint-freq 125000 \
  --value-warmup-steps 200000 \
  --pedestrians 1 \
  --encounter-prob 0.7 \
  --decision-layer-prob 0.7 \
  --subgoal-lookahead 1.5 \
  --subgoal-offset 0.6 \
  --lane-hold 8 15 \
  --goal-distance-range 1 25 \
  --eval-random 40 \
  --n-envs 8 \
  --seed 0
