#!/usr/bin/env bash
#SBATCH --job-name=order_reacher
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_reacher_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_reacher_common.sh
sha256sum "$PROJECT/scripts/run_reacher_stage_a.py" "$PROJECT/order_jepa/core.py"
"$PY" "$PROJECT/scripts/run_reacher_stage_a.py" \
  --stable-worldmodel-source "$SWM_SOURCE" \
  --stablewm-home "$STAGE0_ROOT" \
  --provenance "$PROVENANCE" \
  --out-dir "$REACHER_ROOT/scores" \
  --anchor-count "${ORDER_ANCHORS:-200}" --anchors-per-episode 8 \
  --num-shards 8 --shard-index "${SLURM_ARRAY_TASK_ID:?}"
