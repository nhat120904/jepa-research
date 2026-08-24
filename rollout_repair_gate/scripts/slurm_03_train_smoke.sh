#!/usr/bin/env bash
#SBATCH --job-name=rrg_train_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_train_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/train_predictor.py" \
  --arm multistep_offpolicy --seed 11 --steps 5 --log-every 1 \
  --offpolicy-dir "$PROJECT/outputs/stage1/intermediates" \
  --expert-dir "$PROJECT/outputs/stage1/expert_cache" \
  --out-dir "$PROJECT/outputs/stage1/checkpoints_smoke"

