#!/usr/bin/env bash
#SBATCH --job-name=jepa_covsel_a
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/oracle_covsel_analysis_%j.out
# Submit with afterok dependency on the coverage-selection array.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

TAGS=(
  dino_push_stateprobe dino_push_l2 dino_pick_stateprobe dino_pick_l2 dino_reach_l2
  jepa_push_stateprobe jepa_push_l2 jepa_pick_stateprobe jepa_pick_l2 jepa_reach_l2
)
ITERATIONS=()
CANDIDATES=()
for tag in "${TAGS[@]}"; do
  ITERATIONS+=("results/oracle_covsel_${tag}_iterations.csv")
  CANDIDATES+=("results/oracle_covsel_${tag}_candidates.csv.gz")
done

"$PY" scripts/52_analyze_coverage_selection.py \
  --iterations "${ITERATIONS[@]}" \
  --candidates "${CANDIDATES[@]}" \
  --out-prefix results/oracle_coverage_selection

echo "COVERAGE_SELECTION_ANALYSIS_DONE $(date)"
