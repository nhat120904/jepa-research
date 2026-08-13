#!/usr/bin/env bash
#SBATCH --job-name=ogb_mr_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_matched_refit_aggregate_%A.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
REQUIRE=${OGB_MR_REQUIRE:-32}

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum scripts/85_ogb_matched_refit_aggregate.py

"$STAGE0_ROOT/.venv/bin/python" scripts/85_ogb_matched_refit_aggregate.py \
  --shard-root results/ogb_matched_refit/locked_shards \
  --require-shards "$REQUIRE" \
  --out-dir results/ogb_matched_refit/locked

sha256sum \
  results/ogb_matched_refit/locked/summary.json \
  results/ogb_matched_refit/locked/snapshot_deltas.csv

echo "===== OGB_MATCHED_REFIT_AGGREGATE_DONE ===== $(date -u +%FT%TZ)"
