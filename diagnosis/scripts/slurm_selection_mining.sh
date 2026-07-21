#!/usr/bin/env bash
#SBATCH --job-name=jepa_sel_mine
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/selection_mining_%j.out
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

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/59_mine_selection_populations.py \
  --config configs/diagnostic_metaworld.yaml \
  --model dino_wm_metaworld --task mw-push \
  --episodes 8 --seed0 62000 \
  --horizon 6 --num-act-stepped 3 --max-episode-steps 100 \
  --cem-num-samples 100 --cem-iterations 6 --elite-frac 0.1 --var0 1.0 \
  --top-proxy 10 --top-true 10 --random-count 10 \
  --probe checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
  --ee-probe checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
  --out results/selection_populations_dino_push.pt
echo "SELECTION_MINING_DONE $(date)"
