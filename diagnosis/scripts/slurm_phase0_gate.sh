#!/usr/bin/env bash
#SBATCH --job-name=jepa_phase0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase0_gate_%j.out
# Phase-0 DECISIVE GATE (cheapest-first). Latent oracle = PERFECT latent dynamics
# through the real frozen encoder; the ONLY thing swapped is the COST. The oracle
# ladder localised the contact wall to the L2-in-latent cost, so this asks: does a
# non-L2 cost cross success UNDER perfect dynamics, before we spend any train/
# closed-loop budget?
#   0a l2   : the null (expected push/pick 0/16).
#   0b gobj : object-readout cost g(z_fin)->g(z_goal), pure-readout (gamma_l2=0).
# Decision:
#   gobj flips push/pick 0 -> >0  => cost is viable; green-light the corrector +
#                                    closed-loop (slurm_grounded_corrector.sh) and
#                                    train the learned metric (scripts/33).
#   gobj stays 0/16              => the wall is deeper than the cost (readout/encoder
#                                    precision); pivot to the representation fallback.
# (0c metric needs the trained d_theta from scripts/33 — run that gate separately.)
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
PROBE=${P0_PROBE:-checkpoints/spatial_object_probe_${M}.pt}
EPISODES=${P0_EPISODES:-16}
TASKS=${P0_TASKS:-"mw-reach mw-push mw-pick-place"}
BETA=${P0_BETA:-1.0}; GAMMA=${P0_GAMMA:-0.0}   # pure object-readout cost by default

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M probe=$PROBE tasks=[$TASKS] episodes=$EPISODES beta=$BETA gamma_l2=$GAMMA"
ls -la "$PROBE"

set -e
# 0a — L2 null
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost l2 --tasks $TASKS --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_l2.csv

# 0b — object-readout cost (the decisive test of readout precision under perfect dynamics)
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost gobj --probe "$PROBE" --beta "$BETA" --gamma-l2 "$GAMMA" \
    --tasks $TASKS --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_gobj.csv

echo "===== PHASE0_GATE_DONE ====="; date
echo "--- 0a l2 ---";   cat results/latent_oracle_l2.csv
echo "--- 0b gobj ---"; cat results/latent_oracle_gobj.csv
