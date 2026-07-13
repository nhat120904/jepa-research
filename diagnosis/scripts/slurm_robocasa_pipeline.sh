#!/usr/bin/env bash
#SBATCH --job-name=jepa_robocasa
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G   # 1B ViT-G latents already cached; only extract small dino_wm/jepa_wm. Fits under 256G/user QOS.
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/robocasa_pipe_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
# CRITICAL: the RoboCasaDataset custom-teleop loader walks $JEPAWM_DSET/robocasa/
# for *im256.hdf5; without this the prior run raised (caught) and produced 0
# transitions ("insufficient_data" everywhere). Point it at diagnosis/data.
export JEPAWM_DSET="$PWD/data"
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export CAI_JEPA_ALLOW_HEAVY_MODEL=1
export PATH="$PWD/.venv/bin:$PATH"
CFG=configs/diagnostic_robocasa.yaml
PY=.venv/bin/python
# Default to the non-gated small predictor (dino_wm_droid); override with
# ROBOCASA_MODEL. jepa_wm_droid needs gated DINOv3 weights; vjepa2_ac is 1B.
export CAI_JEPA_ONLY_MODEL="${ROBOCASA_MODEL:-dino_wm_droid}"
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) model=$CAI_JEPA_ONLY_MODEL JEPAWM_DSET=$JEPAWM_DSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set -e
echo "===== STEP 03 extract latents ($CAI_JEPA_ONLY_MODEL, 7-dim Franka) ====="; date
$PY scripts/03_extract_latents.py --config $CFG
echo "===== STEP 04 classify regimes ====="; date
$PY scripts/04_classify_regimes.py --config $CFG
echo "===== STEP 05 run diagnostic ====="; date
$PY scripts/05_run_diagnostic.py --config $CFG
echo "===== STEP 12 boundary diagnostic ====="; date
$PY scripts/12_boundary_diagnostic.py --config $CFG
echo "===== ROBOCASA_PIPELINE_DONE ====="; date
