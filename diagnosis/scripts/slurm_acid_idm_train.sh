#!/usr/bin/env bash
#SBATCH --job-name=jepa_acid_idm
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acid_idm_%A_%a.out
#
# Train one held-out-split ACID-style inverse verifier per MetaWorld checkpoint.
# Submit (do not run Python on the login node):
#   sbatch scripts/slurm_acid_idm_train.sh
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

MODELS=(dino_wm_metaworld jepa_wm_metaworld)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:?submit as array 0-1}]}
CFG=configs/diagnostic_metaworld.yaml
MANIFEST=checkpoints/splits/acid_${MODEL}_split0.json
OUT=checkpoints/acid_idm_${MODEL}_split0.pt

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID} $(date)"
echo "model=$MODEL manifest=$MANIFEST out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

.venv/bin/python scripts/51_train_acid_idm.py \
  --config "$CFG" --model "$MODEL" \
  --split-seed 0 --train-seed 0 --split-manifest "$MANIFEST" \
  --epochs "${ACID_EPOCHS:-20}" --max-train "${ACID_MAX_TRAIN:-30000}" \
  --max-val "${ACID_MAX_VAL:-5000}" --max-test "${ACID_MAX_TEST:-5000}" \
  --out "$OUT"

echo "===== ACID_IDM_DONE ===== $(date)"
