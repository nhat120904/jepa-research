#!/usr/bin/env bash
#SBATCH --job-name=spwm_enc
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_enc_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SPLIT=${SPLIT:?submit with SPLIT=train|val}
RUN_ID=${RUN_ID:-lewm_20260904}
SEED=${SEED:-0}
sha256sum "$PROJECT/scene_lewm.py" "$PROJECT/scripts/encode_cache_latents.py" \
          "$PROJECT/scripts/slurm_encode_latents.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/encode_cache_latents.py" \
  --checkpoint "$CACHE_ROOT/checkpoints/$RUN_ID/seed$SEED/lewm_best.pt" \
  --cache-dir "$CACHE_ROOT/cache/$SPLIT" \
  --batch-size 512
echo "DONE $(date -u +%FT%TZ)"
