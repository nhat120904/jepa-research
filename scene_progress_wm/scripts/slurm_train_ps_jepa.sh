#!/usr/bin/env bash
#SBATCH --job-name=spwm_psjepa
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_psjepa_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh

ARM=${ARM:?submit with ARM=ps_jepa|ps_frame|ps_action_history|ps_noaction_pred}
SEED=${SEED:-0}
STEPS=${STEPS:-12000}
BATCH_SIZE=${BATCH_SIZE:-32}
SEGMENT_BLOCKS=${SEGMENT_BLOCKS:-48}
BURN_IN=${BURN_IN:-16}
PREDICT_BLOCKS=${PREDICT_BLOCKS:-5}
BELIEF_DIM=${BELIEF_DIM:-256}
RUN_ID=${RUN_ID:-ps_jepa_20260905}

sha256sum "$PROJECT/ps_jepa.py" \
          "$PROJECT/scripts/train_ps_jepa.py" \
          "$PROJECT/scripts/slurm_train_ps_jepa.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" "$PROJECT/scripts/train_ps_jepa.py" \
  --arm "$ARM" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --segment-blocks "$SEGMENT_BLOCKS" \
  --burn-in "$BURN_IN" \
  --predict-blocks "$PREDICT_BLOCKS" \
  --belief-dim "$BELIEF_DIM" \
  --cache-dir "$CACHE_ROOT/cache/train" \
  --val-cache-dir "$CACHE_ROOT/cache/val" \
  --out-dir "$CACHE_ROOT/checkpoints/$RUN_ID"

echo "DONE $(date -u +%FT%TZ)"
