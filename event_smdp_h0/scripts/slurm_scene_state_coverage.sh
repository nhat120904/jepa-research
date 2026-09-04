#!/usr/bin/env bash
#SBATCH --job-name=scene_state_coverage
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_state_coverage_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_ID=${RUN_ID:-confirmatory_20260904}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_event_history.py" "$PROJECT/scene_history_dataset.py" \
  "$PROJECT/scripts/diagnose_scene_state_coverage.py"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/diagnose_scene_state_coverage.py" \
  --data-root "$PROJECT/outputs/scene_h1/data" \
  --replication-root \
    "$PROJECT/outputs/scene_event_perception_replication/eval/${RUN_ID}" \
  --out-dir "$PROJECT/outputs/scene_state_coverage/${RUN_ID}"
