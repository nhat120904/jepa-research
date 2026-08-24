#!/usr/bin/env bash
#SBATCH --job-name=crod_h0_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/crod_h0_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/crod_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
cd "$REPO"
sha256sum "$PROJECT/scripts/analyze_h0.py" "$PROJECT/scripts/slurm_h0_aggregate.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_h0.py" \
  --shards "$PROJECT/outputs/h0/shards" \
  --out-dir "$PROJECT/outputs/h0/aggregate"
