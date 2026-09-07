#!/usr/bin/env bash
#SBATCH --job-name=order_rsmoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_rsmoke_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_reacher_common.sh
sha256sum "$PROJECT/scripts/run_reacher_stage_a.py" "$PROJECT/order_jepa/core.py"
"$PY" "$PROJECT/tests/test_core.py"
"$PY" "$PROJECT/scripts/run_reacher_stage_a.py" \
  --stable-worldmodel-source "$SWM_SOURCE" \
  --stablewm-home "$STAGE0_ROOT" \
  --provenance "$PROVENANCE" \
  --out-dir "$REACHER_ROOT/smoke/scores" \
  --anchor-count 4 --anchors-per-episode 1 --num-shards 1 --shard-index 0
