#!/usr/bin/env bash
#SBATCH --job-name=jepa_covsel
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/oracle_covsel_%A_%a.out
# Coverage-vs-selection audit: two checkpoints, both contact tasks, l2/stateprobe,
# plus reach×l2 honest controls. Candidate populations are simulator rolled and
# encoder scored, so this must run on GPU compute nodes.
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
EPISODES=${COVSEL_EPISODES:-16}
SEED0=${COVSEL_SEED0:-40000}

MODELS=(
  dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld
  jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld
)
TASKS=(
  mw-push mw-push mw-pick-place mw-pick-place mw-reach
  mw-push mw-push mw-pick-place mw-pick-place mw-reach
)
COSTS=(
  stateprobe l2 stateprobe l2 l2
  stateprobe l2 stateprobe l2 l2
)
TAGS=(
  dino_push_stateprobe dino_push_l2 dino_pick_stateprobe dino_pick_l2 dino_reach_l2
  jepa_push_stateprobe jepa_push_l2 jepa_pick_stateprobe jepa_pick_l2 jepa_reach_l2
)

I=${SLURM_ARRAY_TASK_ID:?submit this script as its declared array}
MODEL=${MODELS[$I]}
TASK=${TASKS[$I]}
COST=${COSTS[$I]}
TAG=${TAGS[$I]}
PREFIX=results/oracle_covsel_${TAG}

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} model=$MODEL task=$TASK cost=$COST episodes=$EPISODES seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

EXTRA=()
if [[ "$COST" == "stateprobe" ]]; then
  EXTRA+=(
    --probe "checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt"
    --ee-probe "checkpoints/ee_probe_${MODEL}_offpolicy.pt"
  )
fi

"$PY" scripts/51_oracle_coverage_selection.py \
  --config "$CFG" --model "$MODEL" --tasks "$TASK" --cost "$COST" \
  --episodes "$EPISODES" --seed0 "$SEED0" --strict-success --dump-candidates \
  --out-prefix "$PREFIX" "${EXTRA[@]}"

echo "COVERAGE_SELECTION_CELL_DONE tag=$TAG $(date)"
