#!/usr/bin/env bash
#SBATCH --job-name=jepa_trmfit
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/trm_train_%A_%a.out
#
# Two frozen encoders x three head seeds.  Each model reuses one immutable
# trajectory-level 70/15/15 manifest; only train trajectories update the head.
# Do not execute the Python command directly on the login node.
set -euo pipefail

cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_TORCH_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
IDX=${SLURM_ARRAY_TASK_ID:?submit as array 0-5}
MODELS=(dino_wm_metaworld jepa_wm_metaworld)
MODEL_IDX=$((IDX / 3))
HEAD_SEED=$((IDX % 3))
MODEL=${MODELS[$MODEL_IDX]}
SPLIT_SEED=${TRM_SPLIT_SEED:-0}
MANIFEST=checkpoints/splits/trm_${MODEL}_split${SPLIT_SEED}.json
OUT=checkpoints/trm_${MODEL}_s${HEAD_SEED}.pt

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "TRM train model=$MODEL head_seed=$HEAD_SEED manifest=$MANIFEST out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" scripts/51_train_trm.py --config "$CFG" --model "$MODEL" \
  --split-manifest "$MANIFEST" --split-seed "$SPLIT_SEED" \
  --val-frac 0.15 --test-frac 0.15 \
  --planning-horizon 6 --frameskip 5 --label-scale 224 \
  --epochs "${TRM_EPOCHS:-10}" --pairs-per-epoch "${TRM_PAIRS_PER_EPOCH:-10000}" \
  --eval-pairs "${TRM_EVAL_PAIRS:-4096}" --batch-size "${TRM_BATCH_SIZE:-128}" \
  --seed "$HEAD_SEED" --out "$OUT"

echo "===== TRM_TRAIN_DONE model=$MODEL head_seed=$HEAD_SEED ===== $(date)"
