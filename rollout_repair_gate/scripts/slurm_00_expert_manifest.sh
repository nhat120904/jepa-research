#!/usr/bin/env bash
#SBATCH --job-name=rrg_manifest
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_manifest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
cd "$REPO"
sha256sum "$PROJECT/PROTOCOL.md" "$PROJECT/scripts/prepare_expert_manifest.py"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/prepare_expert_manifest.py" \
  --offpolicy-manifest "$REPO/physical_search_distillation/outputs/h0/manifest.json" \
  --num-sequences 15360 \
  --out "$PROJECT/outputs/stage1/expert_manifest.json"

