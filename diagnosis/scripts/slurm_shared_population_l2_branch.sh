#!/usr/bin/env bash
#SBATCH --job-name=jepa_l2branch
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/l2_branch_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python scripts/54_shared_population_branch.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --tasks mw-push \
  --probe checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
  --ee-probe checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
  --branches l2 true_state --carrier true_state \
  --episodes 8 --seed0 42000 \
  --out-prefix results/shared_branch_l2_dino_push
