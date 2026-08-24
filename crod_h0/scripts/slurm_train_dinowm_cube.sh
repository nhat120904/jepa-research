#!/usr/bin/env bash
#SBATCH --job-name=crod_train_dino
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/crod_train_dino_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/crod_h0"
SWM="$REPO/diagnosis/external/stable-worldmodel"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

cd "$SWM"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum scripts/train/prejepa.py scripts/train/config/prejepa.yaml "$PROJECT/scripts/slurm_train_dinowm_cube.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" scripts/train/prejepa.py \
  output_model_name=crod_dinowm_cube_seed42 \
  subdir=crod_dinowm_cube_seed42 \
  seed=42 \
  dataset_name=ogbench/cube_single_expert.h5 \
  batch_size=128 \
  num_workers=16 \
  '~wm.encoding.proprio' \
  trainer.max_epochs=10 \
  trainer.devices=1 \
  trainer.strategy=auto
test -s "$STAGE0_ROOT/checkpoints/crod_dinowm_cube_seed42/config.json"
test -s "$STAGE0_ROOT/checkpoints/crod_dinowm_cube_seed42/weights_epoch_10.pt"
