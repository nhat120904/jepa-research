#!/usr/bin/env bash
#SBATCH --job-name=crod_comp_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/crod_comp_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/crod_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
cd "$REPO"
sha256sum "$PROJECT/scripts/analyze_complementarity.py" "$PROJECT/scripts/slurm_complementarity_aggregate.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_complementarity.py" \
  --shards "$PROJECT/outputs/complementarity/shards" \
  --out-dir "$PROJECT/outputs/complementarity/aggregate"
