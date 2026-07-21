#!/usr/bin/env bash
#SBATCH --job-name=jepa_factcost_a
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/factorized_cost_analysis_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

EPISODES=${FACTORIZED_EPISODES:-16}
SEED0=${FACTORIZED_SEED0:-61000}
.venv/bin/python scripts/58_analyze_factorized_cost_ladder.py \
  --episodes "$EPISODES" --seed0 "$SEED0" \
  --out-prefix results/factorized_cost_ladder_pilot
