#!/usr/bin/env bash
#SBATCH --job-name=jepa_presel_an
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cem_preselection_analysis_%j.out
# Large candidate CSV analysis is kept off the login node.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

TAGS=(
  dino_push_stateprobe dino_push_l2 dino_pick_stateprobe dino_pick_l2
  jepa_push_stateprobe jepa_push_l2 jepa_pick_stateprobe jepa_pick_l2
)
CANDIDATES=()
for tag in "${TAGS[@]}"; do
  CANDIDATES+=("results/cem_preselection_${tag}_candidates.csv.gz")
done

.venv/bin/python scripts/53_analyze_cem_preselection.py \
  --candidates "${CANDIDATES[@]}" \
  --out-prefix results/cem_preselection_audit --n-bootstrap 5000

echo "CEM_PRESELECTION_ANALYSIS_DONE $(date)"
