#!/usr/bin/env bash
#SBATCH --job-name=ew_wmrepair
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$REPAIR_SCRIPT" --parent "$PARENT_RUN" --run "$REPAIR_RUN" --family "$1"
