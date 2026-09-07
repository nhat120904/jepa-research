#!/usr/bin/env bash
#SBATCH --job-name=acm_gate1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_gate1_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/gate1_ranking.py"
for M in gate0:lewm gate0_dino:dino_wm; do
  SRC=${M%%:*}; TAG=${M##*:}
  echo "=== $TAG ==="
  "$PY" "$PROJECT/scripts/gate1_ranking.py" \
    --root "$PROJECT/outputs/$SRC" --label "$TAG" \
    --out "$PROJECT/outputs/gate1_${TAG}.json"
done
