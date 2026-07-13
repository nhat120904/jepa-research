#!/usr/bin/env bash
#SBATCH --job-name=jepa_phase0b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase0_gate_v2_%j.out
# Phase-0 gate, finishing run. v1 (job 23267) timed out on the MIG slice (~4.5
# min/episode); it DID finish 0a l2 (reach 16/16, push 0/16, pick 0/16 — the known
# null) and a partial 0b gobj with gamma_l2=0 (push 0/15, obj ~22cm). This run tests
# the ACTUAL closed-loop object cost config (gamma_l2=1, beta=5 — same as
# slurm_grounded_corrector) on the contact tasks only, fewer episodes so it finishes.
# l2 is NOT rerun (results/latent_oracle_l2.csv already complete).
#   gobj flips push/pick 0 -> >0  => cost viable under perfect dynamics; green-light
#                                    Track A corrector + Track B metric.
#   gobj stays 0                  => with PERFECT dynamics + the real cost config the
#                                    PROBE-object cost still can't plan contact while
#                                    the TRUE-object state-oracle gets 16/16 => the
#                                    readout/representation precision is the wall
#                                    (Phase 3), not the predictor or the cost formula.
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
EPISODES=${P0_EPISODES:-8}
TASKS=${P0_TASKS:-"mw-push mw-pick-place"}
BETA=${P0_BETA:-5.0}; GAMMA=${P0_GAMMA:-1.0}    # the real closed-loop object-cost config

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M probe=$PROBE tasks=[$TASKS] episodes=$EPISODES beta=$BETA gamma_l2=$GAMMA"
ls -la "$PROBE"

set -e
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost gobj --probe "$PROBE" --beta "$BETA" --gamma-l2 "$GAMMA" \
    --tasks $TASKS --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_gobj_g1b5.csv
echo "===== PHASE0_GATE_V2_DONE ====="; date
cat results/latent_oracle_gobj_g1b5.csv
