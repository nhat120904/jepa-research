#!/usr/bin/env bash
#SBATCH --job-name=ew_diagref
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# PRIVILEGED diagnostic reference tables for the 2026-10-05 component audit (no training, no simulator).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/diag_20261005/refs_${SLURM_JOB_ID}
mkdir -p "$OUT"; cp "$PROJECT/docs/diag_20261005/build_refs.py" "$OUT/"
"$OGB_PY" "$PROJECT/docs/diag_20261005/build_refs.py" "$OUT"
