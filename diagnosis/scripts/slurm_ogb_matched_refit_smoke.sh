#!/usr/bin/env bash
#SBATCH --job-name=ogb_mr_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_matched_refit_smoke_%A.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT=${1:-0}
ITERS=${2:-6}
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} SNAPSHOT=$SNAPSHOT ITERS=$ITERS $(date -u +%FT%TZ)"
sha256sum scripts/84_ogb_matched_refit.py scripts/slurm_ogb_matched_refit_smoke.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/84_ogb_matched_refit.py \
  --snapshot-index "$SNAPSHOT" \
  --cem-iterations "$ITERS" \
  --provenance-artifacts "$STAGE0_ROOT/artifacts/audit_locked_array" \
  --provenance-shards results/ogb_true_endpoint_corrected/locked_shards \
  --out-dir "results/ogb_matched_refit/smoke_${SNAPSHOT}_${ITERS}"

echo "===== OGB_MATCHED_REFIT_SMOKE_DONE snapshot=$SNAPSHOT ===== $(date -u +%FT%TZ)"
