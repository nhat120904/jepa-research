#!/usr/bin/env bash
#SBATCH --job-name=mwm_g3_prep
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g3_prep_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/prepare_control_features.py" \
          "$PROJECT/scripts/slurm_gate3_prepare.sh"
"$PY" "$PROJECT/scripts/prepare_control_features.py" \
  --dataset "$PROJECT/outputs/gate3_full/control.pt" \
  --out "$PROJECT/outputs/gate3_full/features.pt" \
  --batch-size 48 --feature-dim 128 --pca-samples 12000 \
  --eval-episodes 128 --device cuda
