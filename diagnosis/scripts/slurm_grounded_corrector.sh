#!/usr/bin/env bash
#SBATCH --job-name=jepa_gcorr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/grounded_corrector_%j.out
# Post-oracle-ladder fix: object-grounded residual corrector INSIDE the CEM unroll.
# The latent oracle (scripts/30) proved the L2-in-latent cost is the contact wall;
# this trains Δ(z,a) so g(corrected ẑ_t) tracks the object through the rollout
# (scripts/31, gated on cf-corr + K-step obj-MSE), then runs the closed-loop sweep
# with the gobjc arm against the l2/hdyn baselines (paired, same env+seeds, strict).
#   gobjc success_end > 0 on push/pick  -> the in-loop object channel flips contact
#   gobjc == 0 like hdyn/l2             -> latent geometry still binds; needs a
#                                          repr-level fix, not a corrector
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
M=${GC_MODEL:-dino_wm_metaworld}
PROBE=checkpoints/spatial_object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
RES=checkpoints/grounded_corrector_${M}.pt
OUT=${GC_OUT:-results/metaworld_grounded_corrector.csv}
EPISODES=${GC_EPISODES:-16}
BETA=${GC_BETA:-5.0}; GAMMA=${GC_GAMMA:-1.0}
TASKS=${GC_TASKS:-"mw-reach mw-push mw-pick-place"}
ARMS=${GC_ARMS:-"l2 hdyn gobjc"}
LAMBDA_OBJ=${GC_LAMBDA_OBJ:-10}; LAMBDA_CF=${GC_LAMBDA_CF:-5}; LAMBDA_MS=${GC_LAMBDA_MS:-5}
K=${GC_K:-3}; EPOCHS=${GC_EPOCHS:-6}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M arms=[$ARMS] tasks=[$TASKS] beta=$BETA gamma_l2=$GAMMA out=$OUT"
ls -la "$PROBE" "$DYN"

set -e
# 1) train the grounded corrector (writes $RES; prints the cf-corr + obj-MSE gates)
$PY scripts/31_train_grounded_corrector.py \
    --config "$CFG" --model "$M" --probe "$PROBE" --dyn-head "$DYN" \
    --epochs "$EPOCHS" --lambda-obj "$LAMBDA_OBJ" --lambda-cf "$LAMBDA_CF" \
    --lambda-ms "$LAMBDA_MS" --K "$K"

# 2) closed-loop sweep — gobjc against the l2/hdyn baselines, paired + strict
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --probe "$PROBE" --dyn-head "$DYN" --residual-head "$RES" \
    --beta "$BETA" --gamma-l2 "$GAMMA" \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" --out "$OUT" --strict-success
echo "===== GROUNDED_CORRECTOR_DONE ====="; date
cat "$OUT"
