#!/usr/bin/env bash
#SBATCH --job-name=gfpr_train_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/gfpr_train_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/gfpr_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/train_fold.py" "$PROJECT/scripts/slurm_train_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/train_fold.py" \
  --fold 0 \
  --ensemble-size 2 \
  --epochs 1 \
  --out-dir "$PROJECT/outputs/fold_smoke"

