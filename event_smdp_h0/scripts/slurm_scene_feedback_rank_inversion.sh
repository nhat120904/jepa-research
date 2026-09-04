#!/usr/bin/env bash
#SBATCH --job-name=scene_rank_inv
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_rank_inv_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_ID=${RUN_ID:?submit with RUN_ID}
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RUN=$RUN_ID $(date -u +%FT%TZ)"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/contrast_scene_feedback_rank_inversion.py" \
  --summary "$PROJECT/outputs/scene_feedback_robustness/aggregate/${RUN_ID}/summary.json" \
  --out-dir "$PROJECT/outputs/scene_feedback_robustness/aggregate/${RUN_ID}"
