#!/usr/bin/env bash
#SBATCH --job-name=jepa_loramsk
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/lora_msk_train_%j.out
# GT-SUPERVISED LoRA (the candidate "solution") — slurm_lora_train.sh distills the
# COUNTERFACTUAL from the dyn-head teacher, which is itself 0/16 on contact, so that
# arm is teacher-ceilinged. Here we turn ON the teacher-FREE masked-object term
# (λ_msk>0, scripts/26 object_saliency_mask): the object's own tokens are masked in
# the context and the rollout must recover obj_{t+1} from the GROUND-TRUTH 39-dim
# state (OBJECT_SLICE) — no teacher, so NOT ceilinged. This is the only lever that
# can move contact closed-loop above the baseline. Writes a DISTINCT checkpoint
# (predictor_lora_msk_*) so the distill-only LoRA stays available for the ablation.
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
M=${LORA_MODEL:-dino_wm_metaworld}
SPATIAL=checkpoints/spatial_object_probe_${M}.pt
POOLED=checkpoints/object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
CKPT=checkpoints/predictor_lora_msk_${M}.pt
EPOCHS=${LORA_EPOCHS:-8}; RANK=${LORA_RANK:-8}; ALPHA=${LORA_ALPHA:-16}
# GT-supervised term ON; keep a light cf-distill + recon so dynamics stay consistent.
LOBJ=${LORA_LOBJ:-10}; LCF=${LORA_LCF:-5}
LMSK=${LORA_LMSK:-10}; MTOPK=${LORA_MTOPK:-12}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "lambda_msk=$LMSK mask_topk=$MTOPK lambda_cf=$LCF lambda_obj=$LOBJ -> $CKPT"

set -e
echo "===== STEP 1/3: train GT-supervised predictor LoRA (scripts/26, λ_msk=$LMSK) ====="; date
$PY scripts/26_train_predictor_lora_cf.py --config $CFG --model $M \
    --spatial-probe $SPATIAL --dyn-head $DYN \
    --epochs $EPOCHS --rank $RANK --alpha $ALPHA --lambda-obj $LOBJ --lambda-cf $LCF \
    --lambda-msk $LMSK --mask-topk $MTOPK --out $CKPT
ls -la $CKPT

echo "===== STEP 2/3: GATE 1 — V3-spatial corr on the GT-LoRA predictor (scripts/24) ====="; date
$PY scripts/24_counterfactual_spatial.py --config $CFG --model $M \
    --pooled-probe $POOLED --spatial-probe $SPATIAL \
    --predictor-lora $CKPT --out results/counterfactual_spatial_lora_msk.csv

echo "===== STEP 3/3: GATE 2 — rollout fidelity on the GT-LoRA predictor (scripts/23) ====="; date
$PY scripts/23_rollout_fidelity.py --config $CFG --model $M \
    --probe $SPATIAL --predictor-lora $CKPT --H 6 \
    --out results/rollout_fidelity_lora_msk.csv
echo "===== LORA_MSK_TRAIN_DONE ====="; date
echo "--- GATE 1 (cf-corr) ---"; cat results/counterfactual_spatial_lora_msk.csv
echo "--- GATE 2 (factual tracking) ---"; cat results/rollout_fidelity_lora_msk.csv
