#!/usr/bin/env bash
#SBATCH --job-name=spwm_swap
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_swap_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh

RUN_ID=${RUN_ID:-ps_jepa_20260905}
SEED=${SEED:-0}
SAMPLES=${SAMPLES:-20000}
ARMS=${ARMS:-ps_jepa,ps_frame}

sha256sum "$PROJECT/ps_jepa.py" "$PROJECT/scripts/history_swap_audit.py" \
          "$PROJECT/scripts/slurm_history_swap.sh"

"$PY" "$PROJECT/scripts/history_swap_audit.py" \
  --cache-dir "$CACHE_ROOT/cache/val" \
  --checkpoint-root "$CACHE_ROOT/checkpoints/$RUN_ID" \
  --out-dir "$PROJECT/outputs/history_swap/diagnostic/$RUN_ID" \
  --arms "$ARMS" --seed "$SEED" --samples "$SAMPLES"

echo "DONE $(date -u +%FT%TZ)"
