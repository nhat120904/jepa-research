#!/usr/bin/env bash
#SBATCH --job-name=lscope_b02_cal
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/stage_b_restore_cal_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
CONFIG="$PROJECT_ROOT/latent_scope_20260909/configs/stage_b_restore_calibration.json"
POLICY_VENV="$RUNTIME_ROOT/policy_venv"
SIM_VENV="$RUNTIME_ROOT/sim_venv"
GROOT_SRC="$RUNTIME_ROOT/src/Isaac-GR00T"
ROBOCASA_SRC="$SHARED_ROOT/src/robocasa365"
ROBOSUITE_SRC="$SHARED_ROOT/src/robosuite"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/stage_b_restore_cal_$SLURM_JOB_ID"
PORT=$((20000 + SLURM_JOB_ID % 20000))

mkdir -p "$OUTPUT_DIR"
export HF_HOME="$RUNTIME_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MESA_SHADER_CACHE_DIR="$RUNTIME_ROOT/mesa_shader_cache"
mkdir -p "$MESA_SHADER_CACHE_DIR"

SERVER_LOG="$OUTPUT_DIR/policy_server.log"
SERVER_STATE="$OUTPUT_DIR/policy_server_state.json"
env PYTHONPATH="$GROOT_SRC" \
    "$POLICY_VENV/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/run_groot_policy_server.py" \
    --config "$CONFIG" \
    --state-output "$SERVER_STATE" \
    --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909/scripts:$ROBOCASA_SRC:$ROBOSUITE_SRC" \
    "$SIM_VENV/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/calibrate_stage_b_event_restore.py" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --server-pid "$SERVER_PID" \
    --server-state "$SERVER_STATE" \
    --port "$PORT"
