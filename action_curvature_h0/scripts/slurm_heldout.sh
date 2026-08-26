#!/usr/bin/env bash
#SBATCH --job-name=acm_held
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_held_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TASK_ID=${SLURM_ARRAY_TASK_ID:?}

# 4 arms x 4 snapshot chunks.  Arms: original, then the three already-trained
# continuation seeds (lambda_AS = 0).  No AS arm reaches held-out: it failed the
# manipulation check on dev.
ARMS=(original lam0_seed0 lam0_seed1 lam0_seed2)
ARM=${ARMS[$((TASK_ID / 4))]}
CHUNK=$((TASK_ID % 4))
if [ "$ARM" = "original" ]; then EXTRA=""
else EXTRA="--state-dict $PROJECT/outputs/dev/${ARM}/checkpoint.pt"; fi
echo "arm=$ARM chunk=$CHUNK"

# Orders 64-127, 16 snapshots per chunk, identical directions and sigmas for
# every arm so the paired per-snapshot comparison is clean.
START=$((64 + CHUNK * 16))
for OFFSET in $(seq 0 15); do
  SNAP=$((START + OFFSET))
  "$PY" "$PROJECT/scripts/measure_curvature.py" \
    --snapshot-index "$SNAP" --n-snapshots 128 \
    --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
    --action-source dataset --horizons 5 \
    --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
    --n-directions 2 --repeats 2 --model-dtype float64 $EXTRA \
    --out-dir "$PROJECT/outputs/heldout/${ARM}/snapshot_$(printf '%03d' "$SNAP")"
done
echo "HELDOUT $ARM chunk $CHUNK COMPLETE $(date -u +%FT%TZ)"
