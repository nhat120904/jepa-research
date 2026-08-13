#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
sha256sum "$DIAG/scripts/71_ogb_stage0_smoke.py" "$DIAG/scripts/slurm_ogb_stage0_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/71_ogb_stage0_smoke.py \
  --out results/ogb_stage0_smoke.json

echo "===== OGB_STAGE0_SMOKE_WRAPPER_DONE ===== $(date -u +%FT%TZ)"
