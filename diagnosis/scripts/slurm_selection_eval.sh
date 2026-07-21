#!/usr/bin/env bash
#SBATCH --job-name=jepa_sel_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=16:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/selection_eval_%A_%a.out
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

I=${SLURM_ARRAY_TASK_ID:?submit as array}
ARM=$((I / 3))
SEED=$((I % 3))
TAGS=(lora_tail last_regression last_pairwise last_tail)
TAG=${TAGS[$ARM]}
CHECKPOINT="checkpoints/selection_${TAG}_s${SEED}.pt"
OUT="results/selection_eval_${TAG}_s${SEED}.csv"

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} index=$I tag=$TAG seed=$SEED $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/61_eval_selection_encoder.py \
  --config configs/diagnostic_metaworld.yaml --checkpoint "$CHECKPOINT" \
  --tasks mw-push mw-reach --episodes 16 --seed0 63000 \
  --horizon 6 --num-act-stepped 3 --max-episode-steps 100 \
  --cem-num-samples 100 --cem-iterations 6 --elite-frac 0.1 --var0 1.0 \
  --out "$OUT"
echo "SELECTION_EVAL_DONE tag=$TAG seed=$SEED $(date)"
