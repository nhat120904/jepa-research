#!/usr/bin/env bash
#SBATCH --job-name=ew_uneval
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$RUN_DIR/source:$RUN_DIR"
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$RUN_DIR/run_eval.py" --root "$RUN_DIR" --family "$1" --training-job "$2" --seed "$3"
