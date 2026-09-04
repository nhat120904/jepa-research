#!/usr/bin/env bash
#SBATCH --job-name=event_scene_agg
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/event_scene_agg_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
PILOT_JOB_ID=${PILOT_JOB_ID:?submit with --export=ALL,PILOT_JOB_ID=<array_job>}
INPUT="$PROJECT/outputs/scene_gate0/pilot/shards"
OUT="$PROJECT/outputs/scene_gate0/pilot/aggregate/${PILOT_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} PILOT=$PILOT_JOB_ID $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/scripts/analyze_scene_gate0.py" \
  "$PROJECT/scripts/slurm_scene_gate0_analysis.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_gate0.py" \
  --input-root "$INPUT" --expected 32 --bootstrap 20000 --out-dir "$OUT"

