#!/usr/bin/env bash
#SBATCH --job-name=jepa_paper_render
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/paper_render_%j.out
set -euo pipefail
PAPER=/home/nhatnc129/nhat.nc/jepa-research/paper
mkdir -p "$PAPER/rendered"
rm -f "$PAPER"/rendered/page-*.png
pdftoppm -png -r 120 "$PAPER/main.pdf" "$PAPER/rendered/page"
find "$PAPER/rendered" -maxdepth 1 -type f -name 'page-*.png' -printf '%f %s\n' | sort
echo "===== PAPER_RENDER_DONE ====="
