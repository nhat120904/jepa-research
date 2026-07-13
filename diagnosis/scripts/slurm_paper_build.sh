#!/usr/bin/env bash
#SBATCH --job-name=jepa_paper
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/paper_build_%j.out
# Compile on a compute node so even document builds do not consume login-node CPU.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf | grep -E '^(Pages|Page size)'
echo "===== PAPER_BUILD_DONE ====="
