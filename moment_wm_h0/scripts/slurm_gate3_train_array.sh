#!/usr/bin/env bash
#SBATCH --job-name=mwm_g3_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g3_train_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ARMS=(mse cov_mse cadm)
TASK=${SLURM_ARRAY_TASK_ID:?}
ARM=${ARMS[$((TASK / 3))]}
SEED=$((TASK % 3))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK ARM=$ARM SEED=$SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/train_strong_baseline.py" \
          "$PROJECT/scripts/slurm_gate3_train_array.sh"
"$PY" "$PROJECT/scripts/train_strong_baseline.py" \
  --features "$PROJECT/outputs/gate3_full/features.pt" \
  --arm "$ARM" --seed "$SEED" \
  --out-dir "$PROJECT/outputs/gate3_full/models/$ARM/seed_$SEED" \
  --epochs 60 --patience 10 --batch-size 128 --device cuda
