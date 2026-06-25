#!/usr/bin/env bash
#SBATCH --job-name=jepa_optdS
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/option_d_strong_%j.out
# Option D, STRONGER corrector: push gate-1 cf-corr past 0.5 (v037 reached +0.37,
# cf-gap still falling at epoch 8). Triple the counterfactual distillation weight
# (λ_cf 10->30) and train longer (8->12 epochs). Writes _strong artifacts so it
# does NOT clobber the v037 checkpoint the parallel closed-loop job is reading.
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
CKPT=checkpoints/cf_predictor_${M}_strong.pt

EPOCHS=${OPTD_EPOCHS:-12}; LOBJ=${OPTD_LOBJ:-10}; LCF=${OPTD_LCF:-30}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "===== STEP 1/3: train STRONG cf-predictor (λ_cf=$LCF, epochs=$EPOCHS) ====="; date
$PY scripts/25_train_cf_predictor.py --config $CFG --model $M \
    --spatial-probe $SPATIAL --dyn-head $DYN \
    --epochs $EPOCHS --lambda-obj $LOBJ --lambda-cf $LCF --out $CKPT
ls -la $CKPT

echo "===== STEP 2/3: GATE 1 — V3-spatial (scripts/24) ====="; date
$PY scripts/24_counterfactual_spatial.py --config $CFG --model $M \
    --pooled-probe $POOLED --spatial-probe $SPATIAL \
    --residual-head $CKPT --out results/counterfactual_spatial_cf_strong.csv

echo "===== STEP 3/3: GATE 2 — rollout fidelity (scripts/23) ====="; date
$PY scripts/23_rollout_fidelity.py --config $CFG --model $M \
    --probe $SPATIAL --residual-head $CKPT --H 6 \
    --out results/rollout_fidelity_cf_strong.csv

echo "===== OPTION_D_STRONG_DONE ====="; date
echo "--- GATE 1 strong (target corr>=0.5) ---"; cat results/counterfactual_spatial_cf_strong.csv
echo "--- GATE 2 strong (must not regress ~3cm) ---"; cat results/rollout_fidelity_cf_strong.csv
