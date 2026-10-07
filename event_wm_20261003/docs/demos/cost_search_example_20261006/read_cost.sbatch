#!/usr/bin/env bash
#SBATCH --job-name=ew_costdemo
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$EXAMPLE_DIR/source"
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$EXAMPLE_DIR/read_cost.py"
