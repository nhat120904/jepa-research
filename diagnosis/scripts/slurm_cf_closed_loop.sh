#!/usr/bin/env bash
#SBATCH --job-name=jepa_cfcl
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_closed_loop_%j.out
# Closed-loop with the option-D corrected predictor F+Δ (the +0.37 cf-corr ckpt,
# snapshot _v037). Does reviving the counterfactual object channel flip contact
# success? Arms (paired, same env+seeds):
#   l2     — frozen predictor, L2 cost              (baseline, reproduces prior 0/16)
#   l2c    — CORRECTED predictor F+Δ, L2 cost       (does the fixed rollout alone help?)
#   hdync  — CORRECTED predictor + grounded object cost (β object-dominant)  (full stack)
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
M=dino_wm_metaworld
PROBE=checkpoints/spatial_object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
RES=checkpoints/cf_predictor_${M}_v037.pt
OUT=${CF_OUT:-results/metaworld_cf_closed_loop.csv}
EPISODES=${CF_EPISODES:-16}; BETA=${CF_BETA:-5.0}
TASKS=${CF_TASKS:-"mw-push mw-pick-place"}; ARMS=${CF_ARMS:-"l2 l2c hdync"}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "residual-head=$RES  beta=$BETA  arms=[$ARMS]  tasks=[$TASKS]  out=$OUT"
ls -la "$RES" "$PROBE" "$DYN"

set -e
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --probe "$PROBE" --dyn-head "$DYN" --residual-head "$RES" --beta "$BETA" \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" --out "$OUT"
echo "===== CF_CLOSED_LOOP_DONE ====="; date
cat "$OUT"
