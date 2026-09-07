#!/usr/bin/env bash
#SBATCH --job-name=mwm_g3_data
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g3_data_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/generate_control_dataset.py" \
          "$PROJECT/scripts/slurm_gate3_generate.sh"
"$PY" "$PROJECT/scripts/generate_control_dataset.py" \
  --out "$PROJECT/outputs/gate3_full/control.pt" \
  --episodes 2400 --queries 12 --batch-size 64 --device cuda
