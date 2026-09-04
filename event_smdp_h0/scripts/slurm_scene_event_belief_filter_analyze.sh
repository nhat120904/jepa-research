#!/usr/bin/env bash
#SBATCH --job-name=scene_event_belief_filter_ana
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_belief_filter_ana_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
EVAL_JOB_ID=${EVAL_JOB_ID:?submit with EVAL_JOB_ID}
OUT="$PROJECT/outputs/scene_event_belief_filter/aggregate/${EVAL_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} EVAL=$EVAL_JOB_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_scene_event_belief_filter.py" \
  "$PROJECT/scripts/slurm_scene_event_belief_filter_analyze.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_event_belief_filter.py" \
  --input-root "$PROJECT/outputs/scene_event_belief_filter/eval/${EVAL_JOB_ID}" \
  --expected-shards 32 --out-dir "$OUT"
