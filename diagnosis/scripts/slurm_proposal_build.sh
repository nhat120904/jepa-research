#!/usr/bin/env bash
#SBATCH --job-name=jepa_proposal
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/proposal_build_%j.out
# Compile on a compute node so even document builds do not consume login-node CPU.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/proposal
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf | grep -E '^(Pages|Page size)'
# The proposal is a tables-only document: this listing must stay empty.
echo "--- embedded images (must be empty) ---"
pdfimages -list main.pdf || true
echo "===== PROPOSAL_BUILD_DONE ====="
