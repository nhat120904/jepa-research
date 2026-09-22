#!/usr/bin/env bash
#SBATCH --job-name=pa_review_default
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/review_default_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 1
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
SOURCE=/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_53529
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
export PYTHONPATH="$SOURCE/code:$SOURCE/deps"
export PA_REVIEW_OUTPUT=/mnt/data/nhatnc129/jepa/predictive_abstraction/review_default_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
"$PY" "$PROJECT/scripts/check_default_headroom.py"
cp "$PROJECT/scripts/check_default_headroom.py" "$PROJECT/scripts/slurm_review_default.sh" "$PA_REVIEW_OUTPUT/"
