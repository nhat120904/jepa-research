#!/usr/bin/env bash
#SBATCH --job-name=crod_manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/crod_manifest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/crod_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
cd "$REPO"
sha256sum "$PROJECT/scripts/prepare_manifest.py" "$PROJECT/scripts/slurm_manifest.sh"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/prepare_manifest.py" \
  --exclude-manifest "$REPO/diagnosis/results/ogb_stage0/audit_locked/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1a/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1av2/manifest.json" \
  --out "$PROJECT/outputs/h0/manifest.json"
