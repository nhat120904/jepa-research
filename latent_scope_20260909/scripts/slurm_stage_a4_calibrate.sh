#!/usr/bin/env bash
#SBATCH --job-name=lscope_a4_calibrate
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/a4_calibrate_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
SIM_VENV="$RUNTIME_ROOT/sim_venv"
ROBOCASA_SRC="$SHARED_ROOT/src/robocasa365"
ROBOSUITE_SRC="$SHARED_ROOT/src/robosuite"

: "${A4_COLLECTION_JOB_ID:?Submit with A4_COLLECTION_JOB_ID exported}"
COLLECTION_DIR="$RUNTIME_ROOT/outputs/a4_collect_$A4_COLLECTION_JOB_ID"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/a4_calibrate_$SLURM_JOB_ID"

test -f "$COLLECTION_DIR/collection_result.json"
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MESA_SHADER_CACHE_DIR="$RUNTIME_ROOT/mesa_shader_cache"
mkdir -p "$MESA_SHADER_CACHE_DIR"

env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909/scripts:$ROBOCASA_SRC:$ROBOSUITE_SRC" \
    "$SIM_VENV/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/run_stage_a4_calibrate.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_a4.json" \
    --collection-dir "$COLLECTION_DIR" \
    --output-dir "$OUTPUT_DIR"
