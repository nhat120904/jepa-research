#!/usr/bin/env bash
#SBATCH --job-name=acm_g0dino
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=64
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_g0dino_%A_%a.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} SNAPSHOT=${SLURM_ARRAY_TASK_ID:?} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/gate0_collect.py"

"$PY" "$PROJECT/scripts/gate0_collect.py" \
  --snapshot-index "${SLURM_ARRAY_TASK_ID}" \
  --manifest "$PERD/outputs/h0/manifest.json" \
  --populations-dir "$PROJECT/outputs/cem_populations" \
  --population-index 1 --n-candidates 12 \
  --checkpoint crod_dinowm_cube_seed42/weights_epoch_10.pt --out-dir "$PROJECT/outputs/gate0_dino_smoke"
