#!/usr/bin/env bash
#SBATCH --job-name=lscope_a_setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_stage_a/logs/setup_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
VENV_PATH="$STAGE_ROOT/venv"
ROBOCASA_SRC="$STAGE_ROOT/src/robocasa365"
ROBOSUITE_SRC="$STAGE_ROOT/src/robosuite"
UV_BIN=/home/nhatnc129/.local/bin/uv

mkdir -p "$STAGE_ROOT/logs" "$STAGE_ROOT/outputs" "$STAGE_ROOT/datasets"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    "$UV_BIN" venv "$VENV_PATH" --python 3.11
fi

if ! "$VENV_PATH/bin/python" -c \
    'from importlib.metadata import version; assert version("robosuite") == "1.5.2"; assert version("robocasa") == "1.0.1"; assert version("lerobot") == "0.3.3"'; then
    "$UV_BIN" pip install --python "$VENV_PATH/bin/python" -e "$ROBOSUITE_SRC"
    "$UV_BIN" pip install --python "$VENV_PATH/bin/python" -e "$ROBOCASA_SRC"
fi

export PYTHONPATH="$ROBOCASA_SRC:$ROBOSUITE_SRC:${PYTHONPATH:-}"
"$VENV_PATH/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/prepare_stage_a.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_a.json" \
    --output "$STAGE_ROOT/outputs/setup_result.json"
