#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum scripts/72_ogb_stage0_candidate_audit.py scripts/73_ogb_stage0_aggregate.py
"$STAGE0_ROOT/.venv/bin/python" scripts/73_ogb_stage0_aggregate.py \
  --shards results/ogb_stage0/audit_locked_shards \
  --out results/ogb_stage0/audit_locked \
  --expected 32 \
  --bootstrap 10000 \
  --seed 20260810

echo "===== OGB_STAGE0_AGGREGATE_WRAPPER_DONE ===== $(date -u +%FT%TZ)"
