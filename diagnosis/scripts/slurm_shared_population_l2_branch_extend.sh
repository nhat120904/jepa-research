#!/usr/bin/env bash
#SBATCH --job-name=jepa_l2branch_ext
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/l2_branch_ext_%A_%a.out
# Strengthens Table 6's weakest cell: the latent-L2 cost-only refitting
# intervention was previously n=8 and push-only (results/shared_branch_l2_dino_push*,
# seed0=42000, jobs 38669/38678). Task 0 extends the SAME push cell from 8 to 16
# seeds at the same seed0 (42000-42015), so the original 8 seeds are a strict
# subset and results remain comparable/mergeable. Task 1 adds a NEW pick-place
# cell at a fresh disjoint seed block (44000-44015), never used elsewhere.
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
MODEL=dino_wm_metaworld
OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt

TASKS=(mw-push mw-pick-place)
SEEDS=(42000 44000)
TAGS=(dino_push_n16 dino_pick_n16)
I=${SLURM_ARRAY_TASK_ID:?submit this script as its declared array}
TASK=${TASKS[$I]}
SEED0=${SEEDS[$I]}
TAG=${TAGS[$I]}
EPISODES=16

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} model=$MODEL task=$TASK episodes=$EPISODES seed0=$SEED0 $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" scripts/54_shared_population_branch.py \
  --config "$CFG" --model "$MODEL" --tasks "$TASK" \
  --probe "$OBJ" --ee-probe "$EEP" \
  --branches l2 true_state --carrier true_state \
  --episodes "$EPISODES" --seed0 "$SEED0" \
  --out-prefix "results/shared_branch_l2_${TAG}"

echo "L2_BRANCH_EXTEND_CELL_DONE tag=$TAG $(date)"
