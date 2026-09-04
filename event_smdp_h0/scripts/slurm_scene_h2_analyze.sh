#!/usr/bin/env bash
#SBATCH --job-name=scene_h2_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h2_analyze_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
AUDIT_JOB_ID=${AUDIT_JOB_ID:?submit with --export=ALL,AUDIT_JOB_ID=<array_job>}
OUT="$PROJECT/outputs/scene_h2/aggregate/${AUDIT_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} AUDIT=$AUDIT_JOB_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_scene_h2.py" "$PROJECT/scripts/slurm_scene_h2_analyze.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_h2.py" \
  --input-root "$PROJECT/outputs/scene_h2/shards" --expected-shards 16 \
  --reference-budget 14 --max-budget 112 --out-dir "$OUT"

