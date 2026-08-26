#!/usr/bin/env bash
#SBATCH --job-name=acm_aggregate
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_aggregate_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh

EXPECTED=$(( 64 * 3 ))
FOUND=$(find "$PROJECT/outputs/stage1" -name summary.json | wc -l)
if [ "$FOUND" -ne "$EXPECTED" ]; then
  echo "refusing to aggregate: found $FOUND shards, expected $EXPECTED" >&2
  exit 1
fi

"$PY" "$PROJECT/scripts/aggregate.py" \
  --shard-root "$PROJECT/outputs/stage1" \
  --out-dir "$PROJECT/outputs/stage1/aggregate" --n-resamples 10000
