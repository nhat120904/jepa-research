#!/usr/bin/env bash
#SBATCH --job-name=rrg_fixed_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-31%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_fixed_eval_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT_INDEX=$((SLURM_ARRAY_TASK_ID * 4))
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/evaluate_fixed_pool.py" \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --data-dir "$PROJECT/outputs/stage1/intermediates" \
  --checkpoint-dir "$PROJECT/outputs/stage1/checkpoints" \
  --out-dir "$PROJECT/outputs/stage1/fixed_eval"

