#!/usr/bin/env bash
#SBATCH --job-name=jepa_latoracle
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/latent_oracle_%j.out
# Latent oracle (scripts/30) — separates PREDICTOR from ENCODER. Run AFTER the
# state oracle (scripts/29) returns "model is the gap" on contact. Perfect latent
# dynamics through the REAL frozen encoder (sim-step -> render -> encode), same
# upstream L2-in-latent cost / horizon / strict success as scripts/18; only F is
# swapped out.
#   latent-oracle PASS + WM-l2 FAIL -> PREDICTOR is the bottleneck (license the
#                                       GT-supervised predictor fix, scripts/26)
#   latent-oracle FAIL              -> ENCODER representation / L2-latent cost is
#                                       the bottleneck (predictor fix can't help)
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${LATORACLE_MODEL:-dino_wm_metaworld}
OUT=${LATORACLE_OUT:-results/metaworld_latent_oracle.csv}
EPISODES=${LATORACLE_EPISODES:-16}
TASKS=${LATORACLE_TASKS:-"mw-reach mw-push mw-pick-place"}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M tasks=[$TASKS] episodes=$EPISODES out=$OUT"

set -e
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --tasks $TASKS --episodes "$EPISODES" --out "$OUT" --strict-success
echo "===== LATENT_ORACLE_DONE ====="; date
cat "$OUT"
