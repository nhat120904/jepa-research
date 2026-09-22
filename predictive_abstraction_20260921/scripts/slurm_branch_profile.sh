#!/usr/bin/env bash
#SBATCH --job-name=pa_branch_encode_profile
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/branch_profile_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/predictive_abstraction
DATASET="$ROOT/prepare_53629/dataset"
RUN="$ROOT/branch_profile_${SLURM_JOB_ID}"
[[ -f "$DATASET/manifest.json" && ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/configs" "$PROJECT/docs" "$PROJECT/tests" "$RUN/code/"
cp "$PROJECT/scripts/slurm_branch_profile.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
export PYTHONPATH="$RUN/code:$ROOT/preflight_53529/deps"
export PA_UPSTREAM_ROOT="$ROOT/preflight_53529/upstream"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -m unittest discover -s "$RUN/code/tests" -v
"$PY" -m pa_wm.encode --config "$RUN/code/configs/encode_dinov3.json" \
  --dataset-root "$DATASET" --run-dir "$RUN/features"
"$PY" -m pa_wm.train_branches --feature-root "$RUN/features" \
  --output "$RUN/profile" --profile
