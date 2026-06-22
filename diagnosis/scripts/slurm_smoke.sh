#!/usr/bin/env bash
#SBATCH --job-name=jepa_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/smoke_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export PATH="$PWD/.venv/bin:$PATH"
echo "HOST $(hostname)  GPU=$CUDA_VISIBLE_DEVICES  TORCH_HOME=$TORCH_HOME  OSSCKPT=$JEPAWM_OSSCKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== synthetic metric validation (offline) ==="
.venv/bin/python scripts/07_validate_synthetic.py 2>&1 | tail -15
echo "=== smoke_test (loads dino_wm_droid + vjepa2_ac_droid, encode+predict) ==="
.venv/bin/python scripts/smoke_test.py 2>&1 | tail -40
echo "SMOKE_DONE rc=$?"
