#!/usr/bin/env bash
#SBATCH --job-name=jepa_taskcost
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_cost_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python scripts/78_analyze_task_aligned_cost_audit.py \
  --preselection-candidates \
    results/cem_preselection_dino_push_stateprobe_candidates.csv.gz \
    results/cem_preselection_dino_pick_stateprobe_candidates.csv.gz \
    results/cem_preselection_jepa_push_stateprobe_candidates.csv.gz \
    results/cem_preselection_jepa_pick_stateprobe_candidates.csv.gz \
  --branch-candidates \
    results/shared_branch_dino_push_candidates.csv.gz \
    results/shared_branch_dino_pick_candidates.csv.gz \
    results/shared_branch_jepa_push_candidates.csv.gz \
    results/shared_branch_jepa_pick_candidates.csv.gz \
    results/shared_branch_l2_dino_push_candidates.csv.gz \
  --n-permutations 5000 --n-bootstrap 10000 \
  --out-prefix results/task_aligned_cost_audit
