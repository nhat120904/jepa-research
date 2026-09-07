#!/usr/bin/env bash
#SBATCH --job-name=spwm_lewm
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_lewm_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SEED=${SEED:?submit with SEED}
STEPS=${STEPS:-40000}
BATCH_SIZE=${BATCH_SIZE:-128}
TRAIN_TAG=${TRAIN_TAG:-train}
VAL_TAG=${VAL_TAG:-val}
RUN_ID=${RUN_ID:-lewm_20260904}

sha256sum "$PROJECT/scene_lewm.py" \
          "$PROJECT/scripts/train_scene_lewm.py" \
          "$PROJECT/scripts/slurm_train_lewm.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" "$PROJECT/scripts/train_scene_lewm.py" \
  --cache-dir "$CACHE_ROOT/cache/$TRAIN_TAG" \
  --val-cache-dir "$CACHE_ROOT/cache/$VAL_TAG" \
  --out-dir "$CACHE_ROOT/checkpoints/$RUN_ID/seed$SEED" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 8

echo "DONE $(date -u +%FT%TZ)"
