#!/usr/bin/env bash
#SBATCH --job-name=jepa_samestate
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/same_state_%j.out
# Exact same-state causal intervention benchmark.  Every action fan member is
# restored from the same MuJoCo integration snapshot.  GPU is required for the
# frozen encoder/predictor comparison; never run this workload on a login node.
set -euo pipefail
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
export CAI_JEPA_TORCH_THREADS=${CAI_JEPA_TORCH_THREADS:-10}

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
MODELS=${SAMESTATE_MODELS:-"dino_wm_metaworld jepa_wm_metaworld"}
TASKS=${SAMESTATE_TASKS:-"mw-push mw-pick-place"}
EPISODES=${SAMESTATE_EPISODES:-12}
ANCHORS=${SAMESTATE_ANCHORS_PER_EPISODE:-6}
OUT=${SAMESTATE_OUT_PREFIX:-results/metaworld_same_state}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "models=[$MODELS] tasks=[$TASKS] episodes=$EPISODES anchors_per_episode=$ANCHORS"

$PY scripts/49_same_state_intervention.py \
  --config "$CFG" --models $MODELS --tasks $TASKS \
  --episodes "$EPISODES" --anchors-per-episode "$ANCHORS" \
  --horizons 1 2 4 8 --n-candidates 17 \
  --out-prefix "$OUT" --overwrite

echo "===== SAME_STATE_INTERVENTION_DONE ====="; date
