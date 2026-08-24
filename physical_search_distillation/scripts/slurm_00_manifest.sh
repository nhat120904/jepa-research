#!/usr/bin/env bash
#SBATCH --job-name=perd_manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_manifest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
cd "$REPO"
sha256sum "$PROJECT/PROTOCOL.md" "$REPO/counterfactual_flow/scripts/prepare_ogb_phase1a_manifest.py" "$PROJECT/scripts/slurm_00_manifest.sh"
"$STAGE0_ROOT/.venv/bin/python" "$REPO/counterfactual_flow/scripts/prepare_ogb_phase1a_manifest.py" \
  --audit-manifest "$REPO/diagnosis/results/ogb_stage0/audit_locked/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1a/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1av2/manifest.json" \
  --num-snapshots 128 --seed 20260819 \
  --out "$PROJECT/outputs/h0/manifest.json"
