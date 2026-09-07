#!/usr/bin/env bash
#SBATCH --job-name=mwm_g2_extract
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g2_extract_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/extract_anchor_features.py" \
          "$PROJECT/scripts/slurm_gate2_extract.sh"
"$PY" "$PROJECT/scripts/extract_anchor_features.py" \
  --dataset "$PROJECT/outputs/gate2_full/glides.pt" \
  --out "$PROJECT/outputs/gate2_full/anchors.pt" \
  --batch-size 48 --device cuda
