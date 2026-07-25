#!/usr/bin/env bash
#SBATCH --job-name=jepa_presel
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cem_preselection_%A_%a.out
# Full-population, pre-selection audit. Every candidate is simulator rolled and
# encoder scored, so this must run on a GPU compute node.
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
EPISODES=${PRESEL_EPISODES:-16}
SEED0=${PRESEL_SEED0:-41000}

MODELS=(
  dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld
  jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld
)
TASKS=(
  mw-push mw-push mw-pick-place mw-pick-place
  mw-push mw-push mw-pick-place mw-pick-place
)
COSTS=(
  stateprobe l2 stateprobe l2
  stateprobe l2 stateprobe l2
)
TAGS=(
  dino_push_stateprobe dino_push_l2 dino_pick_stateprobe dino_pick_l2
  jepa_push_stateprobe jepa_push_l2 jepa_pick_stateprobe jepa_pick_l2
)

I=${SLURM_ARRAY_TASK_ID:?submit this script as its declared array}
MODEL=${MODELS[$I]}
TASK=${TASKS[$I]}
COST=${COSTS[$I]}
TAG=${TAGS[$I]}
OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt
PREFIX=results/cem_preselection_${TAG}

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} model=$MODEL task=$TASK cost=$COST episodes=$EPISODES seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" scripts/51_oracle_coverage_selection.py \
  --config "$CFG" --model "$MODEL" --tasks "$TASK" --cost "$COST" \
  --probe "$OBJ" --ee-probe "$EEP" \
  --episodes "$EPISODES" --seed0 "$SEED0" --strict-success --dump-candidates \
  --out-prefix "$PREFIX"

echo "CEM_PRESELECTION_CELL_DONE tag=$TAG $(date)"
