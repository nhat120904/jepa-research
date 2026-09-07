#!/usr/bin/env bash
#SBATCH --job-name=mwm_g2_data
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g2_data_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/generate_glide_dataset.py" \
          "$PROJECT/scripts/slurm_gate2_generate.sh"
"$PY" "$PROJECT/scripts/generate_glide_dataset.py" \
  --out "$PROJECT/outputs/gate2_full/glides.pt" \
  --episodes 2400 --batch-size 128 --device cuda
