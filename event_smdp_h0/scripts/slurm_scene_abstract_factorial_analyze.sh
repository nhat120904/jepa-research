#!/usr/bin/env bash
#SBATCH --job-name=scene_abstract_factorial
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_abstract_factorial_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
TERMINAL_JOB_ID=${TERMINAL_JOB_ID:?submit with --export=ALL,TERMINAL_JOB_ID=<array_job>}
OUT="$PROJECT/outputs/scene_abstract_factorial/aggregate/${TERMINAL_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} TERMINAL=$TERMINAL_JOB_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_scene_abstract_factorial.py" \
  "$PROJECT/scripts/slurm_scene_abstract_factorial_analyze.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_scene_abstract_factorial.py" \
  --event-root "$PROJECT/outputs/scene_h2/shards" \
  --terminal-root "$PROJECT/outputs/scene_abstract_factorial/terminal/seed0" \
  --expected-shards 16 --primary-budget 112 --out-dir "$OUT"

