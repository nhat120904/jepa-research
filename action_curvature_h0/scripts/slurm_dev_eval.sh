#!/usr/bin/env bash
#SBATCH --job-name=acm_deveval
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-15%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_deveval_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TASK_ID=${SLURM_ARRAY_TASK_ID:?}

# Task 15 is the original checkpoint (arm 1); 0-14 mirror the sweep's
# lambda/seed layout exactly.
LAMBDAS=(0 0.01 0.03 0.1 0.3)
if [ "$TASK_ID" -eq 15 ]; then
  ARM="original"; EXTRA=""
else
  LAM=${LAMBDAS[$((TASK_ID / 3))]}; SEED=$((TASK_ID % 3))
  ARM="lam${LAM}_seed${SEED}"
  EXTRA="--state-dict $PROJECT/outputs/dev/${ARM}/checkpoint.pt"
fi
echo "evaluating arm=$ARM"

# FIXED diagnostic manifest: identical snapshots, directions and sigmas for
# every arm, so lambda=0 versus AS is a clean comparison.  Guards taken from the
# training log would sit at different sampled sigmas per point and could not be
# compared across arms.
for SNAP in 0 5 10 15 20 25 30 35 40 45 50 55; do
  "$PY" "$PROJECT/scripts/measure_curvature.py" \
    --snapshot-index "$SNAP" --n-snapshots 64 \
    --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
    --action-source dataset --horizons 5 \
    --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
    --n-directions 2 --repeats 2 --model-dtype float64 $EXTRA \
    --out-dir "$PROJECT/outputs/dev_eval/${ARM}/snapshot_$(printf '%03d' "$SNAP")"
done
echo "DEV EVAL $ARM COMPLETE $(date -u +%FT%TZ)"
