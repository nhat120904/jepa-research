#!/usr/bin/env bash
#SBATCH --job-name=order_ranalyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_ranalyze_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_reacher_common.sh
sha256sum "$PROJECT/scripts/analyze_reacher_stage_a.py" "$PROJECT/order_jepa/core.py"
"$PY" "$PROJECT/scripts/analyze_reacher_stage_a.py" \
  --scores-dir "$REACHER_ROOT/scores" \
  --out-dir "$REACHER_ROOT/analysis" \
  --expected-anchors "${ORDER_ANCHORS:-200}" --n-bootstrap 5000
