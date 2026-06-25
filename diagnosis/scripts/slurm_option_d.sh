#!/usr/bin/env bash
#SBATCH --job-name=jepa_optd
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/option_d_%j.out
# Option D — train the predictor's counterfactual object channel, then GATE on the
# diagnostics BEFORE any closed-loop spend (the lesson from option C's illusory null).
#   1. scripts/25  train Δ(z,a): spatial-probe factual + dyn-head counterfactual distillation
#   2. scripts/24  V3-spatial GATE 1 (corrected): cf spread 0.74->~2.4cm, corr -0.015->>=0.5
#   3. scripts/23  rollout-fidelity GATE 2 (corrected): factual tracking must NOT regress (~3cm)
# Cache-only train + GPU gates; no env stepping, so no MUJOCO_GL needed.
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
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=dino_wm_metaworld
SPATIAL=checkpoints/spatial_object_probe_${M}.pt
POOLED=checkpoints/object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
CKPT=checkpoints/cf_predictor_${M}.pt

EPOCHS=${OPTD_EPOCHS:-8}; LOBJ=${OPTD_LOBJ:-10}; LCF=${OPTD_LCF:-10}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "===== STEP 1/3: train cf-predictor (scripts/25) ====="; date
$PY scripts/25_train_cf_predictor.py --config $CFG --model $M \
    --spatial-probe $SPATIAL --dyn-head $DYN \
    --epochs $EPOCHS --lambda-obj $LOBJ --lambda-cf $LCF
ls -la $CKPT

echo "===== STEP 2/3: GATE 1 — V3-spatial on corrected predictor (scripts/24) ====="; date
$PY scripts/24_counterfactual_spatial.py --config $CFG --model $M \
    --pooled-probe $POOLED --spatial-probe $SPATIAL \
    --residual-head $CKPT --out results/counterfactual_spatial_cf.csv

echo "===== STEP 3/3: GATE 2 — rollout fidelity on corrected predictor (scripts/23) ====="; date
$PY scripts/23_rollout_fidelity.py --config $CFG --model $M \
    --probe $SPATIAL --residual-head $CKPT --H 6 \
    --out results/rollout_fidelity_cf.csv

echo "===== OPTION_D_DONE ====="; date
echo "--- GATE 1 (cf-corr; target spread ~2cm, corr>=0.5) ---"; cat results/counterfactual_spatial_cf.csv
echo "--- GATE 2 (factual tracking; must not regress ~3cm) ---"; cat results/rollout_fidelity_cf.csv
