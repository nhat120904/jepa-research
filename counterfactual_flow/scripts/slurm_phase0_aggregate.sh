#!/usr/bin/env bash
#SBATCH --job-name=cf_flow_aggregate
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_aggregate_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/counterfactual_flow"

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_ogb_phase0.py" "$PROJECT/scripts/slurm_phase0_aggregate.sh"
python "$PROJECT/scripts/analyze_ogb_phase0.py" \
  --shards "$PROJECT/outputs/ogbench_cube_phase0/locked_shards" \
  --out-dir "$PROJECT/outputs/ogbench_cube_phase0/locked"
echo "===== CFLOW_PHASE0_AGGREGATE_DONE ===== $(date -u +%FT%TZ)"
