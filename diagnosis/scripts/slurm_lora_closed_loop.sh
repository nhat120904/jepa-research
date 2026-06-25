#!/usr/bin/env bash
#SBATCH --job-name=jepa_loracl
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/lora_closed_loop_%j.out
# 2C falsification matrix (paired, same env+seeds); LoRA toggled per arm on one
# adapter, inverse head seeds the CEM mean for 'inv' arms:
#   l2        — frozen predictor, L2 cost, zero-mean CEM   (baseline, prior 0/16)
#   l2lora    — LoRA-corrected predictor (2A only)         (does fixed rollout alone help?)
#   l2inv     — frozen predictor + inverse seed (2B only)  (does fixed sampling alone help?)
#   l2lorainv — LoRA predictor + inverse seed (2A+2B)      (the composition)
# Records the matrix for the paper even though the gate (corr 0.39 < 0.5) predicts
# null — distillation is ceilinged by the dyn-head teacher (hdyninv already 0/16).
# mw-reach included for the no-harm check.
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
M=${LORA_MODEL:-dino_wm_metaworld}
PROBE=checkpoints/spatial_object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
LORA=checkpoints/predictor_lora_${M}.pt
INV=checkpoints/inverse_proposal_${M}.pt
OUT=${LORA_OUT:-results/metaworld_matrix_closed_loop.csv}
EPISODES=${LORA_EPISODES:-16}; BETA=${LORA_BETA:-5.0}
TASKS=${LORA_TASKS:-"mw-reach mw-push mw-pick-place"}; ARMS=${LORA_ARMS:-"l2 l2lora l2inv l2lorainv"}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "lora=$LORA inv=$INV probe=$PROBE beta=$BETA arms=[$ARMS] tasks=[$TASKS] out=$OUT"
ls -la "$LORA" "$INV" "$PROBE" "$DYN"

set -e
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --probe "$PROBE" --dyn-head "$DYN" --predictor-lora "$LORA" \
    --inverse-head "$INV" --beta "$BETA" \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" --out "$OUT" --strict-success
echo "===== LORA_CLOSED_LOOP_DONE ====="; date
cat "$OUT"
