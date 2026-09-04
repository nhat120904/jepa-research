#!/usr/bin/env bash
#SBATCH --job-name=event_g0_analysis
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/event_g0_analysis_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/analyze_gate0.py" "$PROJECT/scripts/slurm_gate0_analysis.sh"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_gate0.py" \
  --shards "$PROJECT/outputs/gate0/shards" \
  --first-index 1 --expected-shards 64 --primary-budget 64 \
  --out-dir "$PROJECT/outputs/gate0/aggregate"
