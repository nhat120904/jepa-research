#!/usr/bin/env bash
#SBATCH --job-name=jepa_sanity
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/sanity_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
for M in dino_wm_droid vjepa2_ac_droid; do
  echo "########## terver gripper sensitivity: $M ##########"
  $PY scripts/terver_gripper_test.py --config configs/diagnostic_droid.yaml --model $M --max-transitions 512 2>&1 | grep -vE "it/s\]|INFO|WARNING|Dropout|Linear|LayerNorm|Sequential|ModuleList|Block|Attention|FeedForward|RoPE|GELU|Softmax|MLP|norm|proj|head|Identity|^\s*\(|^\s*\)|VisionTransformer|PatchEmbed|Conv3d|Transformer"
done
echo "SANITY_DONE"
