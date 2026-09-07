#!/usr/bin/env bash
#SBATCH --job-name=order_manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_manifest_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/scripts/make_manifest.py" \
  --dataset "$DATASET" --out "$MANIFEST" \
  --anchors "${ORDER_ANCHORS:-200}" --candidates "${ORDER_CANDIDATES:-32}" \
  --order-pairs "${ORDER_PAIRS:-2}" --block-steps 5 --num-hist 3 \
  --seed "${ORDER_SEED:-20260907}"

