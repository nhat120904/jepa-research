#!/usr/bin/env bash
#SBATCH --job-name=lscope_a2_align
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_stage_a/logs/a2_alignment_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
VENV_PATH="$STAGE_ROOT/venv"
ROBOCASA_SRC="$STAGE_ROOT/src/robocasa365"
ROBOSUITE_SRC="$STAGE_ROOT/src/robosuite"

test -f "$STAGE_ROOT/outputs/setup_result.json"
mkdir -p "$STAGE_ROOT/outputs/a2_alignment_$SLURM_JOB_ID"

export PYTHONPATH="$ROBOCASA_SRC:$ROBOSUITE_SRC:${PYTHONPATH:-}"

"$VENV_PATH/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/run_stage_a2_alignment.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_a2.json" \
    --output-dir "$STAGE_ROOT/outputs/a2_alignment_$SLURM_JOB_ID"
