#!/usr/bin/env bash
#SBATCH --job-name=scene_event_perception_diag
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_perception_diag_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_ID=${RUN_ID:-confirmatory_20260904}
OUT="$PROJECT/outputs/scene_event_perception_replication/diagnostic/${RUN_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/diagnose_scene_event_perception_replication.py" \
  "$PROJECT/scripts/slurm_scene_event_perception_diagnose.sh"
"$RUNTIME/.venv/bin/python" \
  "$PROJECT/scripts/diagnose_scene_event_perception_replication.py" \
  --input-root "$PROJECT/outputs/scene_event_perception_replication/eval/${RUN_ID}" \
  --expected-shards 128 --out-dir "$OUT"
