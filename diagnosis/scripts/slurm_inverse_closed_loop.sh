#!/usr/bin/env bash
#SBATCH --job-name=jepa_invcl
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/inverse_closed_loop_%j.out
# Lever #2 closed-loop: does seeding the CEM mean with the inverse proposal FLIP
# contact success? Arms (paired, same env+seeds):
#   l2       — frozen predictor, L2 cost, zero-mean CEM   (baseline, prior 0/16)
#   l2inv    — frozen predictor, L2 cost, INVERSE-SEEDED CEM
#   hdyninv  — frozen predictor, grounded object cost, INVERSE-SEEDED CEM (full stack)
# mw-reach is included for the no-harm check (must stay >= 37.5%/50.0% strict).
# --probe is the SPATIAL object probe: it supplies BOTH the hdyn object cost AND the
# inverse seed's Δobj target g(z_goal)-g(z_t).
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
M=${INV_MODEL:-dino_wm_metaworld}
PROBE=checkpoints/spatial_object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
INV=checkpoints/inverse_proposal_${M}.pt
OUT=${INV_OUT:-results/metaworld_inverse_closed_loop.csv}
EPISODES=${INV_EPISODES:-16}; BETA=${INV_BETA:-5.0}
TASKS=${INV_TASKS:-"mw-reach mw-push mw-pick-place"}; ARMS=${INV_ARMS:-"l2 l2inv hdyninv"}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "inverse-head=$INV  probe=$PROBE  beta=$BETA  arms=[$ARMS]  tasks=[$TASKS]  out=$OUT"
ls -la "$INV" "$PROBE" "$DYN"

set -e
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --probe "$PROBE" --dyn-head "$DYN" --inverse-head "$INV" --beta "$BETA" \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" --out "$OUT" --strict-success
echo "===== INVERSE_CLOSED_LOOP_DONE ====="; date
cat "$OUT"
