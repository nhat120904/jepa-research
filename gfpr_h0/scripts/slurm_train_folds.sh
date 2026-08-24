#!/usr/bin/env bash
#SBATCH --job-name=gfpr_fold
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/gfpr_fold_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/gfpr_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} FOLD=$TASK_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/train_fold.py" "$PROJECT/scripts/slurm_train_folds.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/train_fold.py" \
  --fold "$TASK_ID" \
  --out-dir "$PROJECT/outputs/folds/$TASK_ID"

