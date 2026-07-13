#!/usr/bin/env bash
#SBATCH --job-name=jepa_p0metric
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase0_metric_%j.out
# Phase-0 gate 0c — the LAST untested cost candidate: the learned label-free
# quasimetric d_theta (Track B). Different mechanism from gobj (temporal-distance
# geometry over the whole latent, not the object probe). Step 1 trains d_theta on
# the cache (scripts/33, prints the Spearman gates); step 2 runs the latent oracle
# with --cost metric (perfect latent dynamics, only the cost swapped).
#   metric flips push/pick 0 -> >0  => a label-free cost works under perfect dynamics
#                                      => Track B is alive (also -> DROID).
#   metric stays 0 like l2/gobj     => ALL costs fail under perfect dynamics
#                                      => representation precision is the wall
#                                      (confirms Phase 3), the cost lever is closed.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${P0_MODEL:-dino_wm_metaworld}
MET=checkpoints/latent_metric_${M}.pt
EPISODES=${P0_EPISODES:-8}
TASKS=${P0_TASKS:-"mw-reach mw-push mw-pick-place"}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M metric=$MET tasks=[$TASKS] episodes=$EPISODES"

set -e
# 1) train d_theta on the cache (label-free; prints Spearman gates)
$PY scripts/33_train_latent_metric.py --config "$CFG" --model "$M" --out "$MET"

# 2) latent-oracle gate with the learned metric cost
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost metric --metric-head "$MET" \
    --tasks $TASKS --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_metric.csv
echo "===== PHASE0_METRIC_DONE ====="; date
cat results/latent_oracle_metric.csv
