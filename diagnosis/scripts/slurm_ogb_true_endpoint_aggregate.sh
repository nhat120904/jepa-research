#!/usr/bin/env bash
#SBATCH --job-name=ogb_te_aggregate
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum \
  scripts/77_ogb_true_endpoint_aggregate.py \
  scripts/slurm_ogb_true_endpoint_aggregate.sh

python scripts/77_ogb_true_endpoint_aggregate.py \
  --shards results/ogb_true_endpoint_corrected/locked_shards \
  --out results/ogb_true_endpoint_corrected/locked \
  --expected 32 \
  --bootstrap 10000 \
  --seed 20260811

echo "===== OGB_TRUE_ENDPOINT_AGGREGATE_DONE ===== $(date -u +%FT%TZ)"
