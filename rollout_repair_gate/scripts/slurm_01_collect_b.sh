#!/usr/bin/env bash
#SBATCH --job-name=rrg_collect_b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-63%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_collect_b_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=$((SLURM_ARRAY_TASK_ID + 64))
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/collect_intermediates.py" \
  --snapshot-index "$TASK_ID" \
  --out-dir "$PROJECT/outputs/stage1/intermediates"

