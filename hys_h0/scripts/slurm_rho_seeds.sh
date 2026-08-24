#!/usr/bin/env bash
#SBATCH --job-name=hys_rho2
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_rho2_%A_%a.out
# Replication of the seed-0 result: switch was the only arm with CI-clean positive
# rho_final on push (+0.071), despite matching `none` on every training statistic.
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
S=../hys_h0
GATES=(none switch none switch); SEEDS=(1 1 2 2)
I=${SLURM_ARRAY_TASK_ID}; GATE=${GATES[$I]}; SEED=${SEEDS[$I]}
TAG=dino_push_straight_${GATE}_s${SEED}
echo "HOST=$(hostname) gate=$GATE seed=$SEED $(date)"
.venv/bin/python scripts/51_oracle_coverage_selection.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --tasks mw-push --cost straight --projector $S/outputs/projector_${GATE}_seed${SEED}.pt \
  --episodes 16 --seed0 41000 --strict-success --dump-candidates \
  --out-prefix results/cem_preselection_${TAG}
echo "HYS_RHO2_CELL_DONE tag=$TAG $(date)"
