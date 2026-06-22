#!/usr/bin/env bash
#SBATCH --job-name=jepa_dv3
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/droid_jepawm_%j.out
# Add jepa_wm_droid (JEPA-WM, DINOv3 ViT-L ~300M encoder) — the mid-scale point
# of the DROID scaling curve (22M dino_wm -> 300M jepa_wm -> 1B vjepa2_ac).
# The DINOv3 ViT-L backbone .pth was reconstructed from the HF safetensors mirror
# (scripts/convert_dinov3_hf_to_orig.py) because dl.fbaipublicfiles.com is
# firewalled; it lives at $JEPAWM_OSSCKPT/dinov3/. The DINOv3 repo is cloned to
# ~/dinov3 (hubconf trimmed to backbone exports). Decoder staged under the fbai
# basename in TORCH_HOME/hub/checkpoints; the WM ckpt auto-downloads from HF.
# Extract ONLY jepa_wm_droid (dino/vjepa2_ac/oss caches reused), then 04/05/12
# over the FULL 4-model config -> unified droid_diagnostic.csv / droid_boundary.csv.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export JEPAWM_HOME=/home/nhatnc129          # DinoEncoder loads ${JEPAWM_HOME}/dinov3
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
CFG=configs/diagnostic_droid.yaml
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e
echo "===== STEP 03 extract latents (ONLY jepa_wm_droid) ====="; date
CAI_JEPA_ONLY_MODEL=jepa_wm_droid $PY scripts/03_extract_latents.py --config $CFG
echo "OK03_jepawm"
echo "===== STEP 04 classify regimes (all 4 models) ====="; date
$PY scripts/04_classify_regimes.py --config $CFG && echo "OK04_all"
echo "===== STEP 05 run diagnostic (all 4 models) ====="; date
$PY scripts/05_run_diagnostic.py --config $CFG && echo "OK05_all"
echo "===== STEP 12 boundary diagnostic (all 4 models) ====="; date
$PY scripts/12_boundary_diagnostic.py --config $CFG && echo "OK12_all"
echo "===== DROID_JEPAWM_DONE ====="; date
