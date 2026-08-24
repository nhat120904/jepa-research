#!/usr/bin/env bash
#SBATCH --job-name=hys_esmk
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_esmk_%j.out
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
echo "HOST=$(hostname) $(date)"
.venv/bin/python ../hys_h0/scripts/04_train_encoder_straightener.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --tasks mw-push --gate switch --max-trajs-per-task 4 --epochs 1 --batch-size 2 \
  --out-lora ../hys_h0/outputs/smoke_enc.pt
echo "ESMK_DONE $(date)"
