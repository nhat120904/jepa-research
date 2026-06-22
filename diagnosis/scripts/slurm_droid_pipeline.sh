#!/usr/bin/env bash
#SBATCH --job-name=jepa_droid
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/droid_pipe_%j.out
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
echo "===== STEP 03 extract latents (dino_wm_droid + vjepa2_ac_droid) ====="; date
$PY scripts/03_extract_latents.py --config $CFG
echo "===== STEP 04 classify regimes ====="; date
$PY scripts/04_classify_regimes.py --config $CFG
echo "===== STEP 05 run diagnostic ====="; date
$PY scripts/05_run_diagnostic.py --config $CFG
echo "===== STEP 12 boundary diagnostic ====="; date
$PY scripts/12_boundary_diagnostic.py --config $CFG
echo "===== DROID_PIPELINE_DONE ====="; date
