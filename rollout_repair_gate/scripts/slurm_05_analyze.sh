#!/usr/bin/env bash
#SBATCH --job-name=rrg_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_analyze_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/analyze_gate.py" \
  --fixed-dir "$PROJECT/outputs/stage1/fixed_eval" \
  --fresh-dir "$PROJECT/outputs/stage1/fresh_eval" \
  --checkpoint-dir "$PROJECT/outputs/stage1/checkpoints" \
  --out-dir "$PROJECT/outputs/stage1/analysis"

