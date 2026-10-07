#!/usr/bin/env bash
#SBATCH --job-name=ew_unresume
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$RESUME_SCRIPT" --root "$RUN_DIR" --family "$1" --parent-job "$2"
