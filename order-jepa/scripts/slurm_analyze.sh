#!/usr/bin/env bash
#SBATCH --job-name=order_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_analyze_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/scripts/analyze_stage_a.py" \
  --scores-dir "$SCORES" --out-dir "$RUN_ROOT/analysis" \
  --expected-anchors "${ORDER_ANCHORS:-200}" --n-bootstrap 5000

