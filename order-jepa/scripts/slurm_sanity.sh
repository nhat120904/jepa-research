#!/usr/bin/env bash
#SBATCH --job-name=order_sanity
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_sanity_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/tests/test_core.py"
"$PY" "$PROJECT/scripts/sanity_check.py" --out "$RUN_ROOT/sanity/sanity_results.json"

