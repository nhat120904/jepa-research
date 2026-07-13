#!/usr/bin/env bash
#SBATCH --job-name=jepa_scale_shared
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/shared_scaling_%j.out
# Shared 22M->1B scaling protocol: same physical effect mask and exact same
# negative transition IDs for every checkpoint.  Includes model inference and
# must run on a GPU compute node, never the login node.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_ALLOW_HEAVY_MODEL=1
export CAI_JEPA_TORCH_THREADS=${CAI_JEPA_TORCH_THREADS:-10}

PY=.venv/bin/python
CFG=configs/diagnostic_droid.yaml
MODELS=${SHARED_SCALE_MODELS:-"dino_wm_droid jepa_wm_droid vjepa2_ac_droid vjepa2_ac_oss"}
OUT=${SHARED_SCALE_OUT_PREFIX:-results/droid_shared_scaling}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "models=[$MODELS] out=$OUT"

$PY scripts/50_shared_scaling_protocol.py \
  --config "$CFG" --models $MODELS --reference-model dino_wm_droid \
  --K 16 --pool-size 1024 --effect-quantile 0.5 \
  --out-prefix "$OUT" --overwrite

echo "===== SHARED_SCALING_DONE ====="; date
