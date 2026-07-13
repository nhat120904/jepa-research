#!/usr/bin/env bash
#SBATCH --job-name=jepa_overopt
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/overopt_%j.out
# E0 — cost-overoptimization sweep (docs/plans/2026-07-06-overoptimization-sweep-design.md):
# Goodhart curves for latent planning on the latent oracle (perfect dynamics, learned cost).
# Sweeps CEM budget; records per-iteration proxy-vs-true divergence + elite decode error,
# and across-budget closed-loop outcomes. Parameterized:
#   OVEROPT_TASKS    (default "mw-push")
#   OVEROPT_ITERS    (default "2 6 12 24")
#   OVEROPT_SAMPLES  (default "100")   CEM population sizes (best-of-n pressure axis)
#   OVEROPT_COSTS    (default "stateprobe l2")
#   OVEROPT_EPISODES (default 8)
#   OVEROPT_TAG      (default derived from tasks; suffixes the output CSVs)
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
M=${OVEROPT_MODEL:-dino_wm_metaworld}
TASKS=${OVEROPT_TASKS:-"mw-push"}
ITERS=${OVEROPT_ITERS:-"2 6 12 24"}
SAMPLES=${OVEROPT_SAMPLES:-"100"}
COSTS=${OVEROPT_COSTS:-"stateprobe l2"}
EPISODES=${OVEROPT_EPISODES:-8}
TAG=${OVEROPT_TAG:-$(echo "$TASKS" | tr ' ' '_' | tr -d '-')}
OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M tasks=[$TASKS] iters=[$ITERS] samples=[$SAMPLES] costs=[$COSTS] episodes=$EPISODES tag=$TAG"

set -e
# Smoke: 1 episode / 2 iterations / smallest population / first task — fail fast.
SMOKE_TASK=$(echo "$TASKS" | awk '{print $1}')
SMOKE_N=$(echo "$SAMPLES" | awk '{print $1}')
$PY scripts/41_overoptimization_sweep.py --config "$CFG" --model "$M" \
    --tasks "$SMOKE_TASK" --costs $COSTS --iters-grid 2 --samples-grid "$SMOKE_N" --episodes 1 \
    --probe "$OBJ" --ee-probe "$EEP" --strict-success \
    --out-episodes "results/overopt_smoke_episodes_${TAG}.csv" \
    --out-curves "results/overopt_smoke_curves_${TAG}.csv"
echo "===== OVEROPT_SMOKE_OK ====="

$PY scripts/41_overoptimization_sweep.py --config "$CFG" --model "$M" \
    --tasks $TASKS --costs $COSTS --iters-grid $ITERS --samples-grid $SAMPLES \
    --episodes "$EPISODES" \
    --probe "$OBJ" --ee-probe "$EEP" --strict-success \
    --out-episodes "results/overopt_episodes_${TAG}.csv" \
    --out-curves "results/overopt_curves_${TAG}.csv"
echo "===== OVEROPT_SWEEP_DONE ====="; date
