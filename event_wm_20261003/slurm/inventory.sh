#!/usr/bin/env bash
#SBATCH --job-name=ew_inventory
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Play-data press statistics + evaluation-task observations + privileged scripted-executor check.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/inventory_${SLURM_JOB_ID}
record_source "$OUT"
"$OGB_PY" "$PROJECT/scripts/inventory.py" --data "$DATA" --sizes ${SIZES:-3x3 4x4 4x5 4x6} --out "$OUT"
