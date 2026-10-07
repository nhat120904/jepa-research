#!/usr/bin/env bash
#SBATCH --job-name=ew_puzcause
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DIAG_DIR/source"
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$DIAG_DIR/distance_control.py"
