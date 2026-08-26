#!/usr/bin/env bash
#SBATCH --job-name=acm_viab
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_viab_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/check_start_state_viability.py"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/check_start_state_viability.py" \
  --manifest "$PERD/outputs/h0/manifest.json" \
  --populations-dir "$PROJECT/outputs/cem_populations" \
  --first 64 --last 127 \
  --out "$PROJECT/outputs/cem_viability.json"
