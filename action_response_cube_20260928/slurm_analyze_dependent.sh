#!/usr/bin/env bash
#SBATCH --job-name=ar_cube_report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response_cube/logs/%x_%j.out
set -euo pipefail

PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PARENT_JOB_ID=${PARENT_JOB_ID:?set PARENT_JOB_ID}
ROOT=/mnt/data/nhatnc129/jepa/action_response_cube
RUN_DIR=$(cat "$ROOT/job_${PARENT_JOB_ID}.path")
"$PY" /home/nhatnc129/nhat.nc/jepa-research/action_response_cube_20260928/analyze.py \
  "$RUN_DIR/evaluation.json" --output "$RUN_DIR/paired_analysis.json"
