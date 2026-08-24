#!/usr/bin/env bash
#SBATCH --job-name=gfpr_features
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --array=0-31%4
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/gfpr_features_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/gfpr_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/extract_snapshot_features.py" "$PROJECT/scripts/slurm_feature_array.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/extract_snapshot_features.py" \
  --snapshot-index "$TASK_ID" \
  --out-dir "$PROJECT/outputs/features_v3/$TASK_ID"
