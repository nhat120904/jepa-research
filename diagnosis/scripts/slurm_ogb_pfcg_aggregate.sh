#!/usr/bin/env bash
#SBATCH --job-name=ogb_pfcg_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum scripts/75_ogb_pfcg_aggregate.py scripts/slurm_ogb_pfcg_aggregate.sh
"$STAGE0_ROOT/.venv/bin/python" scripts/75_ogb_pfcg_aggregate.py \
  --shards results/ogb_pfcg/locked_v2_shards \
  --out results/ogb_pfcg/locked_v2 \
  --expected 32 \
  --bootstrap 10000 \
  --seed 20260811

echo "===== OGB_PFCG_AGGREGATE_DONE ===== $(date -u +%FT%TZ)"
