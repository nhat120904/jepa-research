#!/usr/bin/env bash
#SBATCH --job-name=cta_paper_build
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/paper_build_%j.out
# H100: submit with sbatch build.sh; record job ID, then inspect squeue and sacct.
# Local Mac with TeX Live: bash build.sh (no Slurm allocation needed).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p build
latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1
latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error supplement.tex > supplement_build.log 2>&1
cp build/main.pdf main.pdf
cp build/supplement.pdf supplement.pdf
python3 - <<'CHECK'
from pathlib import Path
for name in ("main", "supplement"):
    lines = (Path("build") / (name + ".log")).read_text(errors="replace").splitlines()
    for line in lines:
        if any(term in line for term in ("Overfull", "undefined", "Output written")):
            print(name + ": " + line)
CHECK
