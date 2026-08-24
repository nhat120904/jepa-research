#!/usr/bin/env bash
#SBATCH --job-name=cf_p1a_manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_p1a_manifest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/counterfactual_flow"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
cd "$REPO"
sha256sum "$PROJECT/scripts/prepare_ogb_phase1a_manifest.py" "$PROJECT/scripts/slurm_phase1a_manifest.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/prepare_ogb_phase1a_manifest.py" \
  --audit-manifest "$REPO/diagnosis/results/ogb_stage0/audit_locked/manifest.json" \
  --out "$PROJECT/outputs/ogbench_cube_phase1a/manifest.json"
