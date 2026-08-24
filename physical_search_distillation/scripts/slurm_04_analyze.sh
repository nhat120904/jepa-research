#!/usr/bin/env bash
#SBATCH --job-name=perd_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_analyze_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
cd "$REPO"
sha256sum "$PROJECT/PROTOCOL.md" "$PROJECT/scripts/analyze_h0.py" "$PROJECT/scripts/slurm_04_analyze.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_h0.py" \
  --eval-dir "$PROJECT/outputs/h0/eval" --out "$PROJECT/outputs/h0/decision.json"
