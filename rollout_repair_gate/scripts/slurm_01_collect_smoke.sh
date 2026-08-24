#!/usr/bin/env bash
#SBATCH --job-name=rrg_collect_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_collect_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
sha256sum "$PROJECT/PROTOCOL.md" "$PROJECT/core.py" "$PROJECT/scripts/collect_intermediates.py"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/collect_intermediates.py" \
  --snapshot-index 1 \
  --out-dir "$PROJECT/outputs/stage1/intermediates_smoke"

