#!/usr/bin/env bash
#SBATCH --job-name=lscope_a_replay
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_stage_a/logs/replay_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
VENV_PATH="$STAGE_ROOT/venv"
ROBOCASA_SRC="$STAGE_ROOT/src/robocasa365"
ROBOSUITE_SRC="$STAGE_ROOT/src/robosuite"

test -f "$STAGE_ROOT/outputs/setup_result.json"
mkdir -p "$STAGE_ROOT/outputs/replay_$SLURM_JOB_ID"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$ROBOCASA_SRC:$ROBOSUITE_SRC:${PYTHONPATH:-}"

"$VENV_PATH/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/run_stage_a_preflight.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_a.json" \
    --output-dir "$STAGE_ROOT/outputs/replay_$SLURM_JOB_ID"
