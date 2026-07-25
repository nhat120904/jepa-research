#!/usr/bin/env bash
#SBATCH --job-name=jepa_sharedbr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/shared_branch_%A_%a.out
# Simulator-perfect, shared-noise branched CEM; GPU compute node required.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
EPISODES=${SHARED_BRANCH_EPISODES:-8}
SEED0=${SHARED_BRANCH_SEED0:-42000}

MODELS=(dino_wm_metaworld dino_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld)
TASKS=(mw-push mw-pick-place mw-push mw-pick-place)
TAGS=(dino_push dino_pick jepa_push jepa_pick)
I=${SLURM_ARRAY_TASK_ID:?submit this script as its declared array}
MODEL=${MODELS[$I]}
TASK=${TASKS[$I]}
TAG=${TAGS[$I]}
OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} model=$MODEL task=$TASK episodes=$EPISODES seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" scripts/54_shared_population_branch.py \
  --config "$CFG" --model "$MODEL" --tasks "$TASK" \
  --probe "$OBJ" --ee-probe "$EEP" \
  --branches stateprobe true_state --carrier true_state \
  --episodes "$EPISODES" --seed0 "$SEED0" \
  --out-prefix "results/shared_branch_${TAG}"

echo "SHARED_BRANCH_CELL_DONE tag=$TAG $(date)"
