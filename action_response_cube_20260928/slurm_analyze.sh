#!/usr/bin/env bash
#SBATCH --job-name=ar_cube_analysis
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response_cube/logs/%x_%j.out
set -euo pipefail

PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN_DIR=${RUN_DIR:?set RUN_DIR}
EVAL_NAME=${EVAL_NAME:-evaluation.json}
OUT_NAME=${OUT_NAME:-paired_analysis.json}
"$PY" /home/nhatnc129/nhat.nc/jepa-research/action_response_cube_20260928/analyze.py \
  "$RUN_DIR/$EVAL_NAME" --output "$RUN_DIR/$OUT_NAME"
