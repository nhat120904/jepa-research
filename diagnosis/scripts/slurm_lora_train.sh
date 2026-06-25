#!/usr/bin/env bash
#SBATCH --job-name=jepa_lora
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/lora_train_%j.out
# Model-side Rung A: LoRA partial-unfreeze of the ViTPredictor with the option-D
# counterfactual object objective (scripts/26), then GATE before closed-loop:
#   1. scripts/26  train LoRA inside the predictor (cache materialised to RAM once)
#   2. scripts/24  V3-spatial GATE 1 (corrected): corr target >= 0.5 (beat option-D's 0.37)
#   3. scripts/23  rollout-fidelity GATE 2 (corrected): factual tracking must NOT regress
# Cache-only train + GPU gates; no env stepping, so no MUJOCO_GL.
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
M=${LORA_MODEL:-dino_wm_metaworld}
SPATIAL=checkpoints/spatial_object_probe_${M}.pt
POOLED=checkpoints/object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
CKPT=checkpoints/predictor_lora_${M}.pt
EPOCHS=${LORA_EPOCHS:-8}; RANK=${LORA_RANK:-8}; ALPHA=${LORA_ALPHA:-16}
LOBJ=${LORA_LOBJ:-10}; LCF=${LORA_LCF:-10}
LMSK=${LORA_LMSK:-0}; MTOPK=${LORA_MTOPK:-12}   # teacher-free masked-object term (off by default)

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "===== STEP 1/3: train predictor LoRA (scripts/26) ====="; date
$PY scripts/26_train_predictor_lora_cf.py --config $CFG --model $M \
    --spatial-probe $SPATIAL --dyn-head $DYN \
    --epochs $EPOCHS --rank $RANK --alpha $ALPHA --lambda-obj $LOBJ --lambda-cf $LCF \
    --lambda-msk $LMSK --mask-topk $MTOPK
ls -la $CKPT

echo "===== STEP 2/3: GATE 1 — V3-spatial on LoRA predictor (scripts/24) ====="; date
$PY scripts/24_counterfactual_spatial.py --config $CFG --model $M \
    --pooled-probe $POOLED --spatial-probe $SPATIAL \
    --predictor-lora $CKPT --out results/counterfactual_spatial_lora.csv

echo "===== STEP 3/3: GATE 2 — rollout fidelity on LoRA predictor (scripts/23) ====="; date
$PY scripts/23_rollout_fidelity.py --config $CFG --model $M \
    --probe $SPATIAL --predictor-lora $CKPT --H 6 \
    --out results/rollout_fidelity_lora.csv

echo "===== LORA_TRAIN_DONE ====="; date
echo "--- GATE 1 (cf-corr; target corr>=0.5, spread ~2cm) ---"; cat results/counterfactual_spatial_lora.csv
echo "--- GATE 2 (factual tracking; must not regress) ---"; cat results/rollout_fidelity_lora.csv
