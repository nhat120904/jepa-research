#!/usr/bin/env bash
#SBATCH --job-name=ew_gprep
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$RUN_DIR/source:$RUN_DIR"
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$RUN_DIR/run_prepare_generic.py" --root "$RUN_DIR" ${FAMILIES:+--families $FAMILIES}
