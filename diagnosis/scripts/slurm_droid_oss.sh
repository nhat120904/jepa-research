#!/usr/bin/env bash
#SBATCH --job-name=jepa_oss
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/droid_oss_%j.out
# Add vjepa2_ac_oss (public V-JEPA-2-AC ViT-G, ~1B) to the DROID diagnostic.
# Encoder + decoder are already staged (shared with vjepa2_ac_droid); the WM
# checkpoint vjepa2_ac_oss.pth.tar auto-downloads from HF (facebook/jepa-wms).
# Strategy: extract ONLY the oss latents (caches for dino_wm/vjepa2_ac are
# reused), then run 04/05/12 over the FULL 3-model config so the unified
# droid_diagnostic.csv / droid_boundary.csv carry all three. Bootstrap is
# seeded (default_rng(0)) so the dino/vjepa2_ac rows reproduce identically.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
CFG=configs/diagnostic_droid.yaml
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e
echo "===== STEP 03 extract latents (ONLY vjepa2_ac_oss) ====="; date
CAI_JEPA_ONLY_MODEL=vjepa2_ac_oss $PY scripts/03_extract_latents.py --config $CFG
echo "OK03_oss"
echo "===== STEP 04 classify regimes (all 3 models) ====="; date
$PY scripts/04_classify_regimes.py --config $CFG && echo "OK04_all"
echo "===== STEP 05 run diagnostic (all 3 models) ====="; date
$PY scripts/05_run_diagnostic.py --config $CFG && echo "OK05_all"
echo "===== STEP 12 boundary diagnostic (all 3 models) ====="; date
$PY scripts/12_boundary_diagnostic.py --config $CFG && echo "OK12_all"
echo "===== DROID_OSS_DONE ====="; date
