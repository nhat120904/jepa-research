#!/usr/bin/env bash
#SBATCH --job-name=paper_build
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/paper_build_%j.out
# Usage: sbatch --wait build.sh   (compiles main.pdf in this directory)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1
echo "latexmk exit $?"
grep -E "Warning|Error|undefined|Overfull" main.log | grep -v "Font shape" | head -40
