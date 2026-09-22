#!/usr/bin/env bash
#SBATCH --job-name=pa_branches_demo
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/prepare_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 1
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
SOURCE=/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_53529
RUN=/mnt/data/nhatnc129/jepa/predictive_abstraction/prepare_${SLURM_JOB_ID}
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
[[ ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/docs" "$RUN/code/"
cp "$PROJECT/scripts/slurm_prepare_v2.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
export PYTHONPATH="$RUN/code:$SOURCE/deps" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -m pa_wm.demo --output "$RUN/demo" --upstream "$SOURCE/upstream"
"$PY" -m pa_wm.prepare_branches --output "$RUN/dataset" --upstream "$SOURCE/upstream"
