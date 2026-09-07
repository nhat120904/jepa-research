#!/usr/bin/env bash
# Shared paths for ORDER-JEPA jobs. This file is sourced on compute nodes.
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/order-jepa"
PY="${ORDER_PYTHON:-$REPO/diagnosis/.venv/bin/python}"
DINO_WM_ROOT="${DINO_WM_ROOT:-/mnt/data/nhatnc129/jepa/dino_wm_original}"
MODEL_DIR="${DINO_WM_MODEL_DIR:-/mnt/data/nhatnc129/jepa/dino_wm_official/outputs/pusht}"
DATASET="${ORDER_DATASET:-/mnt/data/nhatnc129/jepa/datasets/pusht_noise/val}"
RUN_ROOT="${ORDER_RUN_ROOT:-/mnt/data/nhatnc129/jepa/order_jepa/stage_a}"
MANIFEST="${ORDER_MANIFEST:-$RUN_ROOT/manifest.json}"
BRANCHES="$RUN_ROOT/branches"
SCORES="$RUN_ROOT/scores"

export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
echo "DINO_WM_ROOT=$DINO_WM_ROOT MODEL_DIR=$MODEL_DIR"
