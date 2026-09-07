#!/usr/bin/env bash
#SBATCH --job-name=spwm_pscheck
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_pscheck_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh

RUN_ID=${RUN_ID:-ps_jepa_20260905}
sha256sum "$PROJECT/ps_jepa.py" "$PROJECT/scripts/check_ps_jepa.py"

"$PY" "$PROJECT/scripts/check_ps_jepa.py" \
  --cache-dir "$CACHE_ROOT/cache/val" \
  --out-dir "$PROJECT/outputs/ps_jepa/diagnostic/$RUN_ID"

echo "--- smoke training, 200 steps ---"
STEPS=200 EVAL_EVERY=100 "$PY" "$PROJECT/scripts/train_ps_jepa.py" \
  --arm ps_jepa --seed 0 --steps 200 --warmup-steps 20 --eval-every 100 \
  --val-batches 4 --batch-size 16 --num-workers 4 \
  --cache-dir "$CACHE_ROOT/cache/train" \
  --val-cache-dir "$CACHE_ROOT/cache/val" \
  --out-dir "$CACHE_ROOT/checkpoints/${RUN_ID}_smoke"

echo "DONE $(date -u +%FT%TZ)"
