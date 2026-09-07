#!/usr/bin/env bash
#SBATCH --job-name=mwm_g4_select
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g4_select_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/select_mmr_lambda.py" \
          "$PROJECT/scripts/slurm_gate4_select.sh"
"$PY" "$PROJECT/scripts/select_mmr_lambda.py" \
  --models-root "$PROJECT/outputs/gate4_full/models" \
  --lambdas 0.1 1 10 --seeds 0 1 2 \
  --out "$PROJECT/outputs/gate4_full/selection.json"
