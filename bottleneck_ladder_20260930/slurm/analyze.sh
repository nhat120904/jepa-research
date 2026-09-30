#!/usr/bin/env bash
#SBATCH --job-name=ladder_an
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/bottleneck_ladder/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT=/mnt/data/nhatnc129/jepa/bottleneck_ladder/${RUN_NAME:?set RUN_NAME}
"$PY" /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/analyze.py --run-dir "$OUT" --out "$OUT/analysis_${SLURM_JOB_ID}.json"
