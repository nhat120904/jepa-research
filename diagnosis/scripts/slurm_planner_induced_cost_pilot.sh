#!/usr/bin/env bash
#SBATCH --job-name=jepa_pi_cost
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/planner_induced_cost_pilot_%j.out
# ======================================================================
# Round-0 screen for ITERATIVE PLANNER-INDUCED COST LEARNING (scripts/64).
#
# The selection-aware ENCODER sprint (scripts/60-62) gated
# STOP_METHOD_DIRECTION: closed-loop push stayed 0-5/16 across all four arms
# (results/selection_sprint_report.md). The variant that sprint never ran is
# the iterative loop — mine, relabel with simulator truth, retrain, re-mine —
# together with an objective that is asymmetric in the direction argmin
# actually exploits (ranking a genuinely bad candidate cheap).
#
# The full loop needs re-mining and is expensive. This job runs the cheap
# NECESSARY CONDITION first, on the sprint's own mined buffer:
#
#   with the encoder FROZEN and every arm given identical cached latents,
#   does the regret-weighted (asymmetric) pairwise objective select better on
#   a HELD-OUT episode than the objectives the sprint already tried?
#
# Arms (all share SelectionCostHead, so only the objective differs):
#   regression   — grouped_regression_huber      (sprint capacity baseline)
#   pairwise     — grouped_pairwise_logistic     (uniform ranking, sprint arm)
#   softmin      — grouped_softmin_regret        (weights by PREDICTED argmin mass)
#   regretw_k3   — grouped_regret_weighted_pairwise, kappa=3   (NEW)
#   regretw_k10  — same, kappa=10                              (NEW)
# kappa=0 recovers `pairwise` exactly, so the new arm is a strict
# generalisation and the contrast isolates the asymmetry, not the family.
#
# READ THE RESULT NARROWLY. This is one round, frozen latents, and one-shot
# selection inside already-mined populations — NOT closed-loop success.
# Phases 0/3/3b already showed frozen-latent readouts fail closed-loop, so a
# pass here does not predict k/16; it only licenses spending GPU on the
# mine -> retrain -> re-mine loop. A fail closes that loop cheaply.
# Power is low by construction: the buffer holds 8 mw-push episodes (336
# populations), so leave-one-episode-out gives 8 clusters.
#
#   sbatch scripts/slurm_planner_induced_cost_pilot.sh
#
# Latents are cached to results/selection_latents_dino_push.pt on the first
# run; re-runs that only change arms/epochs/seeds skip the encoder entirely.
# ======================================================================
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_TORCH_THREADS=${CAI_JEPA_TORCH_THREADS:-10}
PY=.venv/bin/python

BUFFER=${PICOST_BUFFER:-results/selection_populations_dino_push.pt}
MODEL=${PICOST_MODEL:-dino_wm_metaworld}
EPOCHS=${PICOST_EPOCHS:-6}
SEEDS=${PICOST_SEEDS:-"0 1 2"}
BATCH=${PICOST_ENCODE_BATCH:-256}
OUT=${PICOST_OUT:-results/planner_induced_cost_pilot.md}

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
echo "buffer=$BUFFER model=$MODEL epochs=$EPOCHS seeds=$SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY scripts/64_planner_induced_cost_pilot.py \
    --buffer "$BUFFER" \
    --model "$MODEL" \
    --device cuda \
    --encode-batch "$BATCH" \
    --epochs "$EPOCHS" \
    --seeds $SEEDS \
    --out "$OUT"

echo "===== PLANNER_INDUCED_COST_PILOT_DONE ===== $(date)"
