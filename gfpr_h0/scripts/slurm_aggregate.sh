#!/usr/bin/env bash
#SBATCH --job-name=gfpr_aggregate
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/gfpr_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/gfpr_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_h0a.py" "$PROJECT/scripts/slurm_aggregate.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_h0a.py" \
  --out-dir "$PROJECT/outputs/aggregate"

