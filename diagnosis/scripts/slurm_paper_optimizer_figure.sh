#!/usr/bin/env bash
#SBATCH --job-name=jepa_paper_fig
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/paper_optimizer_figure_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
mkdir -p ../paper/figures
.venv/bin/python scripts/64_make_optimizer_shift_figure.py \
  --candidates results/cem_preselection_dino_push_stateprobe_candidates.csv.gz \
  --summary results/stateprobe_cem_validation_summary.csv \
  --out ../paper/figures/optimizer_shift.pdf
echo "===== PAPER_OPTIMIZER_FIGURE_DONE ====="
