#!/usr/bin/env bash
#SBATCH --job-name=cf_p1av2_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_p1av2_aggregate_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/counterfactual_flow"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
cd "$REPO"
sha256sum "$PROJECT/scripts/analyze_ogb_phase1av2_proxy_uncertainty.py" "$PROJECT/scripts/slurm_phase1av2_aggregate.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_ogb_phase1av2_proxy_uncertainty.py" \
  --shards "$PROJECT/outputs/ogbench_cube_phase1av2/shards" \
  --out-dir "$PROJECT/outputs/ogbench_cube_phase1av2/aggregate"
