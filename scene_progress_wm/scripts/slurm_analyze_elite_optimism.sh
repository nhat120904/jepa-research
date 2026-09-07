#!/usr/bin/env bash
#SBATCH --job-name=spwm_elite_ana
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_elite_ana_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh

RUN_ID=${RUN_ID:-elite_optimism_20260904}
PRIMARY_ARM=${PRIMARY_ARM:-latent_l2}

sha256sum "$PROJECT/scripts/analyze_elite_optimism.py" \
          "$PROJECT/docs/SCENE_ELITE_OPTIMISM_PROTOCOL.md"

"$PY" "$PROJECT/scripts/analyze_elite_optimism.py" \
  --result-dir "$PROJECT/outputs/elite_optimism/diagnostic/$RUN_ID" \
  --out-dir "$PROJECT/outputs/elite_optimism/aggregate/$RUN_ID" \
  --primary-arm "$PRIMARY_ARM"

echo "DONE $(date -u +%FT%TZ)"
