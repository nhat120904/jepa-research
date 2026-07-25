#!/usr/bin/env bash
#SBATCH --job-name=jepa_permnull
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/residual_permutation_null_%j.out
# Candidate-dump analysis is intentionally kept off the login node.
set -euo pipefail

cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CANDIDATES=(
  results/cem_preselection_dino_push_stateprobe_candidates.csv.gz
  results/cem_preselection_dino_pick_stateprobe_candidates.csv.gz
  results/cem_preselection_jepa_push_stateprobe_candidates.csv.gz
  results/cem_preselection_jepa_pick_stateprobe_candidates.csv.gz
)

.venv/bin/python scripts/56_analyze_residual_permutation_null.py \
  --candidates "${CANDIDATES[@]}" \
  --n-permutations 1000 \
  --n-bootstrap 5000 \
  --out-prefix results/cem_residual_permutation_null

echo "RESIDUAL_PERMUTATION_NULL_DONE $(date)"
