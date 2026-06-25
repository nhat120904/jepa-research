#!/usr/bin/env bash
#SBATCH --job-name=jepa_franka
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G   # 1B ViT-G latents already cached; only extract small jepa_wm. Fits under 256G/user QOS.
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/franka_pipe_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export CAI_JEPA_ALLOW_HEAVY_MODEL=1
export PATH="$PWD/.venv/bin:$PATH"
CFG=configs/diagnostic_franka_custom.yaml
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e
echo "===== STEP 03 extract latents (+jepa_wm_droid 300M; dino_wm/vjepa2_ac cached) ====="; date
$PY scripts/03_extract_latents.py --config $CFG
echo "===== STEP 04 classify regimes ====="; date
$PY scripts/04_classify_regimes.py --config $CFG
echo "===== STEP 05 run diagnostic ====="; date
$PY scripts/05_run_diagnostic.py --config $CFG
echo "===== STEP 12 boundary diagnostic ====="; date
$PY scripts/12_boundary_diagnostic.py --config $CFG
echo "===== FRANKA_PIPELINE_DONE ====="; date
