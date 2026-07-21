#!/usr/bin/env bash
#SBATCH --job-name=jepa_sel_analysis
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/selection_analysis_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python scripts/62_analyze_selection_sprint.py \
  results/selection_eval_lora_tail_s{0,1,2}.csv \
  results/selection_eval_last_regression_s{0,1,2}.csv \
  results/selection_eval_last_pairwise_s{0,1,2}.csv \
  results/selection_eval_last_tail_s{0,1,2}.csv \
  --out-json results/selection_sprint_report.json \
  --out-md results/selection_sprint_report.md
echo "SELECTION_ANALYSIS_DONE $(date)"
