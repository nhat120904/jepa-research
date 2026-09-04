#!/usr/bin/env bash
#SBATCH --job-name=scene_sweep_ana
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_sweep_ana_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_ID=${RUN_ID:?submit with RUN_ID}
GRID_RUN_ID=${GRID_RUN_ID:-grid_20260904}
EXPECTED_SHARDS=${EXPECTED_SHARDS:?submit with EXPECTED_SHARDS}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_feedback.py" \
  "$PROJECT/scripts/analyze_scene_feedback_robustness.py"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_feedback_robustness.py" \
  --eval-root "$PROJECT/outputs/scene_feedback_robustness/eval/${RUN_ID}" \
  --grid-root "$PROJECT/outputs/scene_state_vs_feedback/eval/${GRID_RUN_ID}" \
  --expected-shards "$EXPECTED_SHARDS" \
  --out-dir "$PROJECT/outputs/scene_feedback_robustness/aggregate/${RUN_ID}"
