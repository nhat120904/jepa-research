#!/usr/bin/env bash
#SBATCH --job-name=jepa_mw
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mw_cache_%j.out
# Build the Metaworld latent cache + regime sidecar for dino_wm_metaworld so the
# notebook's Part 1 (transition table + frame galleries) can run. Only 03+04 are
# needed — metaworld_diagnostic.csv / metaworld_boundary.csv already exist.
# Encoder (DINOv2 ViT-S via torch hub) + decoder (staged under fbai basename) +
# WM ckpt (auto-downloads from HF). Parquet at data/hf_mw/metaworld/data.
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
CFG=configs/diagnostic_metaworld.yaml
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "parquet shards: $(ls data/hf_mw/metaworld/data/*.parquet 2>/dev/null | wc -l)"

set -e
echo "===== STEP 03 extract latents (ONLY dino_wm_metaworld) ====="; date
CAI_JEPA_ONLY_MODEL=dino_wm_metaworld $PY scripts/03_extract_latents.py --config $CFG
echo "OK03_mw"
echo "===== STEP 04 classify regimes ====="; date
CAI_JEPA_ONLY_MODEL=dino_wm_metaworld $PY scripts/04_classify_regimes.py --config $CFG
echo "OK04_mw"
echo "===== MW_CACHE_DONE ====="; date
ls -la data/precomputed_latents/metaworld__dino_wm_metaworld.h5*
