#!/usr/bin/env bash
#SBATCH --job-name=jepa_acid_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acid_eval_%A_%a.out
#
# Eight cells = 2 models x 2 dynamics x 2 tasks.  Each cell evaluates terminal
# and ACID costs inside one process on the same held-out seeds and CEM noise.
# Run only after both IDM training array cells succeed:
#   TRAIN=$(sbatch --parsable scripts/slurm_acid_idm_train.sh)
#   sbatch --dependency=afterok:${TRAIN} scripts/slurm_acid_baseline_eval.sh
set -euo pipefail

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
export CAI_JEPA_TORCH_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PATH="$PWD/.venv/bin:$PATH"

MODELS=(dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld)
DYNAMICS=(learned learned oracle oracle learned learned oracle oracle)
TASKS=(mw-push mw-pick-place mw-push mw-pick-place mw-push mw-pick-place mw-push mw-pick-place)
IDX=${SLURM_ARRAY_TASK_ID:?submit as array 0-7}
MODEL=${MODELS[$IDX]}
DYN=${DYNAMICS[$IDX]}
TASK=${TASKS[$IDX]}
IDM=checkpoints/acid_idm_${MODEL}_split0.pt
SEED0=${ACID_SEED0:-22000}
EPISODES=${ACID_EPISODES:-32}
LAMBDA=${ACID_LAMBDA:-0.05}
OUT=results/acid_${MODEL}_${DYN}_${TASK}_seed${SEED0}_n${EPISODES}.csv

test -f "$IDM"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "model=$MODEL dynamics=$DYN task=$TASK seed0=$SEED0 n=$EPISODES lambda=$LAMBDA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

.venv/bin/python scripts/52_eval_acid_baseline.py \
  --config configs/diagnostic_metaworld.yaml --model "$MODEL" --idm "$IDM" \
  --dynamics "$DYN" --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
  --lambda-acid "$LAMBDA" --cem-num-samples "${ACID_SAMPLES:-100}" \
  --cem-iterations "${ACID_ITERS:-6}" --strict-success --out "$OUT"

echo "===== ACID_EVAL_DONE ===== $(date)"
