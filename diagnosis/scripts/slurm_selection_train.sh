#!/usr/bin/env bash
#SBATCH --job-name=jepa_sel_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/selection_train_%A_%a.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_TORCH_THREADS=4

I=${SLURM_ARRAY_TASK_ID:?submit as array}
ARM=$((I / 3))
SEED=$((I % 3))
ADAPTATIONS=(lora last_blocks last_blocks last_blocks)
OBJECTIVES=(tail regression pairwise tail)
TAGS=(lora_tail last_regression last_pairwise last_tail)
ADAPTATION=${ADAPTATIONS[$ARM]}
OBJECTIVE=${OBJECTIVES[$ARM]}
TAG=${TAGS[$ARM]}
OUT="checkpoints/selection_${TAG}_s${SEED}.pt"

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} index=$I tag=$TAG seed=$SEED $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/60_train_selection_encoder.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --buffer results/selection_populations_dino_push.pt \
  --adaptation "$ADAPTATION" --objective "$OBJECTIVE" --seed "$SEED" \
  --last-blocks 4 --epochs 4 --steps-per-epoch 400 \
  --encoder-lr 1e-5 --head-lr 3e-4 --lambda-anchor 1e-4 \
  --temperature 0.05 --huber-beta 0.05 --min-true-gap 0.005 \
  --out "$OUT"
echo "SELECTION_TRAIN_DONE tag=$TAG seed=$SEED $(date)"
