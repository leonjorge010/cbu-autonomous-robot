#!/bin/bash
# Train tegr_ppo_la15 seed 0 on one CBU Compute Lab RTX 5090.
# Submit from ~/cbu-autonomous-robot on the login node: sbatch lab/tegr_la15_s0.sbatch
#SBATCH --job-name=tegr-la15-s0
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

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # CUDA torch (the laptop requirements file pins the CPU wheel). numpy<2 matches
  # the stack this code was tested with.
  uv pip install 'numpy<2' torch 'stable-baselines3==2.9.0' 'gymnasium==1.3.0' \
    tensorboard pillow scipy pyyaml
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -c 'import torch, sys; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)'

exec python -m cash_rl.train \
  --name tegr_ppo_la15_s0 \
  --map cbu_floor2 \
  --timesteps 3000000 \
  --n-envs 8 \
  --goal-distance-range 1 25 \
  --eval-random 40 \
  --subgoal-lookahead 1.5 \
  --subgoal-offset 0.6 \
  --lane-hold 8 15 \
  --seed 0
