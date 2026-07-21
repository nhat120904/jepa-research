#!/usr/bin/env bash
#SBATCH --job-name=jepa_factcost
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/factorized_cost_%A_%a.out
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
EPISODES=${FACTORIZED_EPISODES:-16}
SEED0=${FACTORIZED_SEED0:-61000}

MODELS=(dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld)
TASKS=(mw-push mw-push mw-push mw-push mw-pick-place mw-pick-place mw-pick-place mw-pick-place mw-push mw-push mw-push mw-push mw-pick-place mw-pick-place mw-pick-place mw-pick-place)
ARMS=(decoded_both true_object true_hand true_both decoded_both true_object true_hand true_both decoded_both true_object true_hand true_both decoded_both true_object true_hand true_both)
MODEL_TAGS=(dino dino dino dino dino dino dino dino jepa jepa jepa jepa jepa jepa jepa jepa)
TASK_TAGS=(push push push push pick pick pick pick push push push push pick pick pick pick)

I=${SLURM_ARRAY_TASK_ID:?submit as the declared array}
MODEL=${MODELS[$I]}
TASK=${TASKS[$I]}
ARM=${ARMS[$I]}
TAG=${MODEL_TAGS[$I]}_${TASK_TAGS[$I]}_${ARM}
OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt
OUT=results/factorized_cost_${TAG}_seed${SEED0}_n${EPISODES}.csv

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} commit=$(git rev-parse HEAD) cell=$I model=$MODEL task=$TASK arm=$ARM episodes=$EPISODES seed0=$SEED0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" scripts/57_factorized_cost_ladder.py \
  --config "$CFG" --model "$MODEL" --task "$TASK" --arm "$ARM" \
  --probe "$OBJ" --ee-probe "$EEP" \
  --episodes "$EPISODES" --seed0 "$SEED0" --strict-success \
  --out "$OUT"

echo "FACTORIZED_COST_CELL_DONE tag=$TAG $(date)"
