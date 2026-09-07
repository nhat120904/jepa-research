#!/usr/bin/env bash
#SBATCH --job-name=spwm_prog
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_prog_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
ARM=${ARM:?submit with ARM}
SEED=${SEED:-0}
STEPS=${STEPS:-12000}
RUN_ID=${RUN_ID:-progress_20260904}
sha256sum "$PROJECT/progress_head.py" "$PROJECT/scripts/train_progress_head.py" \
          "$PROJECT/scripts/slurm_train_progress.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/train_progress_head.py" \
  --cache-dir "$CACHE_ROOT/cache/train" \
  --val-cache-dir "$CACHE_ROOT/cache/val" \
  --out-dir "$CACHE_ROOT/checkpoints/$RUN_ID/$ARM/seed$SEED" \
  --arm "$ARM" \
  --seed "$SEED" \
  --steps "$STEPS"
echo "DONE $(date -u +%FT%TZ)"
