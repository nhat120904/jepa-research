#!/usr/bin/env bash
#SBATCH --job-name=cf_p1a_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/counterfactual_flow"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
sha256sum "$PROJECT/scripts/run_ogb_phase1a_acquisition.py" "$PROJECT/scripts/slurm_phase1a_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/run_ogb_phase1a_acquisition.py" \
  --snapshot-index 0 --manifest "$PROJECT/outputs/ogbench_cube_phase1a/manifest.json" \
  --num-samples 48 --cem-steps 4 --topk 8 --alternatives 8 \
  --out-dir "$PROJECT/outputs/ogbench_cube_phase1a/smoke/0"
