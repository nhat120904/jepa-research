#!/usr/bin/env bash
#SBATCH --job-name=order_collect
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-199%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_collect_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/scripts/collect_pusht_anchor.py" \
  --manifest "$MANIFEST" --anchor-index "${SLURM_ARRAY_TASK_ID:?}" \
  --dino-wm-root "$DINO_WM_ROOT" --out-dir "$BRANCHES" --repeats 2

