#!/usr/bin/env bash
#SBATCH --job-name=rrg_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_train_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
ARMS=(one_step_expert multistep_expert multistep_offpolicy)
SEEDS=(11 23 47)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ARM=${ARMS[$((TASK_ID / 3))]}
SEED=${SEEDS[$((TASK_ID % 3))]}
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/train_predictor.py" \
  --arm "$ARM" --seed "$SEED" --steps 2000 --batch-size 64 \
  --offpolicy-dir "$PROJECT/outputs/stage1/intermediates" \
  --expert-dir "$PROJECT/outputs/stage1/expert_cache" \
  --out-dir "$PROJECT/outputs/stage1/checkpoints"

