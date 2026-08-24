#!/usr/bin/env bash
#SBATCH --job-name=hys_hlth
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH 
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_hlth_%j.out
# Runs after the training array. Re-evaluates every saved LoRA on a proper held-out
# split with trajectory-clustered CIs -- the in-training health() used only ~34 ridge
# samples and its object-decode numbers were unreadable.
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
S=../hys_h0
.venv/bin/python $S/scripts/05_eval_encoder_health.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --ckpts "$S/outputs/enc_*_r16_seed*.pt" \
  --out $S/outputs/ENC_HEALTH.json
echo "HEALTH_DONE $(date)"
