#!/usr/bin/env bash
#SBATCH --job-name=order_phys
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_phys_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/scripts/analyze_physical_branches.py" \
  --branches-dir "$BRANCHES" --out "$RUN_ROOT/physical_smoke.json"
