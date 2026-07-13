#!/usr/bin/env bash
#SBATCH --job-name=jepa_confana
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/confirmatory_analysis_%j.out
# Submit with --dependency=afterok:<confirmatory-array-job-id>.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python scripts/49_analyze_confirmatory.py \
  --seed0 "${CONFIRM_SEED0:-20000}" \
  --episodes "${CONFIRM_EPISODES:-64}"
