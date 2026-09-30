#!/usr/bin/env bash
#SBATCH --job-name=consensus_an
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/consensus_planning/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT=/mnt/data/nhatnc129/jepa/consensus_planning/${RUN_NAME:?set RUN_NAME}
"$PY" /home/nhatnc129/nhat.nc/jepa-research/consensus_planning_20260929/scripts/analyze.py --run-dir "$OUT" --out "$OUT/analysis_${SLURM_JOB_ID}.json"
