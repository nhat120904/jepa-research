#!/usr/bin/env bash
#SBATCH --job-name=mwm_g4_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g4_train_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
LAMBDAS=(0.1 1 10)
TASK=${SLURM_ARRAY_TASK_ID:?}
LAMBDA=${LAMBDAS[$((TASK / 3))]}
SEED=$((TASK % 3))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK LAMBDA=$LAMBDA SEED=$SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/train_kernel_mmr.py" \
          "$PROJECT/scripts/slurm_gate4_train_array.sh"
"$PY" "$PROJECT/scripts/train_kernel_mmr.py" \
  --features "$PROJECT/outputs/gate3_full/features.pt" \
  --mmr-lambda "$LAMBDA" --seed "$SEED" \
  --out-dir "$PROJECT/outputs/gate4_full/models/lambda_${LAMBDA}/seed_$SEED" \
  --epochs 60 --patience 10 --batch-size 128 --device cuda
