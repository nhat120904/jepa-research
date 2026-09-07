#!/usr/bin/env bash
#SBATCH --job-name=spwm_gate
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_gate_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SPLIT=${SPLIT:-val}
RUN_ID=${RUN_ID:-gate_20260904}
sha256sum "$PROJECT/scene_eval.py" "$PROJECT/scene_render.py" \
          "$PROJECT/scripts/harness_gate.py" "$PROJECT/scripts/slurm_harness_gate.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/harness_gate.py" \
  --cache-dir "$CACHE_ROOT/cache/$SPLIT" \
  --out-dir "$PROJECT/outputs/harness_gate/diagnostic/$RUN_ID" \
  --goal-offsets 25 50 100 200 \
  --num-episodes 50 \
  --episode-seed 90100 --screen-trivial
echo "DONE $(date -u +%FT%TZ)"
