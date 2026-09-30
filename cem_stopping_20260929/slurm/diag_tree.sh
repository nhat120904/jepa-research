#!/usr/bin/env bash
#SBATCH --job-name=cemstop_diagtree
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
# Outcome stability across K, proxy ranking of CEM iterates, mean-shift rule.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT="$RUN_ROOT/diag_tree_${SLURM_JOB_ID}"
record_source "$OUT"
"$PY" "$PROJECT/scripts/diag_tree.py" --run-dir "$RUN_ROOT/dev_v1" --out "$OUT/diag_tree.json"
