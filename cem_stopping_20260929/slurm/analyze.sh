#!/usr/bin/env bash
#SBATCH --job-name=cemstop_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
# Usage: RUN_NAME=dev_v1 sbatch slurm/analyze.sh [--split test --frozen FILE]
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
RUN_DIR="$RUN_ROOT/${RUN_NAME:?set RUN_NAME}"
"$PY" "$PROJECT/scripts/analyze.py" --run-dir "$RUN_DIR" \
  --out "$RUN_DIR/analysis_${SLURM_JOB_ID}.json" "$@"
