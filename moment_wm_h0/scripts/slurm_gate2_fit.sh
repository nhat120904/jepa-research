#!/usr/bin/env bash
#SBATCH --job-name=mwm_g2_fit
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g2_fit_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/fit_anchor_certificates.py" \
          "$PROJECT/scripts/slurm_gate2_fit.sh"
"$PY" "$PROJECT/scripts/fit_anchor_certificates.py" \
  --dataset "$PROJECT/outputs/gate2_full/glides.pt" \
  --features "$PROJECT/outputs/gate2_full/anchors.pt" \
  --out-dir "$PROJECT/outputs/gate2_full/certificates" \
  --epochs 60 --patience 10 --batch-size 64 --seeds 0 1 2 \
  --bootstrap 10000 --device cuda
