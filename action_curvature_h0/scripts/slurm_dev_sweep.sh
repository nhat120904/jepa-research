#!/usr/bin/env bash
#SBATCH --job-name=acm_sweep
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_sweep_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
# 5 lambdas x 3 paired seeds.  Seed is the fast axis so a lambda's three seeds
# are adjacent, and seed k of every lambda shares batch order and perturbations.
LAMBDAS=(0 0.01 0.03 0.1 0.3)
LAM=${LAMBDAS[$((TASK_ID / 3))]}
SEED=$((TASK_ID % 3))
echo "arm lambda=$LAM seed=$SEED"
"$PY" "$PROJECT/scripts/train_as.py" \
  --lambda-as "$LAM" --seed "$SEED" \
  --steps 1000 --batch-size 16 --log-every 50 \
  --out-dir "$PROJECT/outputs/dev/lam${LAM}_seed${SEED}"
echo "SWEEP TASK $TASK_ID COMPLETE $(date -u +%FT%TZ)"
