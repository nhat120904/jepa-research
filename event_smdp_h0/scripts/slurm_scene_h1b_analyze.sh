#!/usr/bin/env bash
#SBATCH --job-name=scene_h1b_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h1b_analyze_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
MODEL_SEED=${MODEL_SEED:-0}
EVAL_JOB_ID=${EVAL_JOB_ID:?submit with --export=ALL,EVAL_JOB_ID=<array_job>}
OUT="$PROJECT/outputs/scene_h1b/aggregate/${EVAL_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} EVAL=$EVAL_JOB_ID MODEL_SEED=$MODEL_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_scene_h1b.py" "$PROJECT/scripts/slurm_scene_h1b_analyze.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_h1b.py" \
  --h1b-root "$PROJECT/outputs/scene_h1b/eval/seed${MODEL_SEED}" \
  --h1-root "$PROJECT/outputs/scene_h1/eval/seed${MODEL_SEED}" \
  --expected-shards 16 --primary-budget 28 --out-dir "$OUT"

