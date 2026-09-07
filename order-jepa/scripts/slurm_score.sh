#!/usr/bin/env bash
#SBATCH --job-name=order_score
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_score_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/score_pusht_anchor.py" \
  --branches-dir "$BRANCHES" --anchor-count "${ORDER_ANCHORS:-200}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:?}" --num-shards "${ORDER_SCORE_SHARDS:-4}" \
  --dino-wm-root "$DINO_WM_ROOT" --model-dir "$MODEL_DIR" \
  --device cuda --alpha 1.0 --out-dir "$SCORES"

