#!/usr/bin/env bash
#SBATCH --job-name=jepa_sharedbr_an
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/shared_branch_analysis_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python scripts/55_analyze_shared_population_branch.py \
  --iterations \
    results/shared_branch_dino_push_iterations.csv \
    results/shared_branch_dino_pick_iterations.csv \
    results/shared_branch_jepa_push_iterations.csv \
    results/shared_branch_jepa_pick_iterations.csv \
  --out-prefix results/shared_population_branch_audit --n-bootstrap 5000

echo "SHARED_BRANCH_ANALYSIS_DONE $(date)"
