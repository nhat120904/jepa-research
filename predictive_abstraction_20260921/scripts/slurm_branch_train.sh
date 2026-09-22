#!/usr/bin/env bash
#SBATCH --job-name=pa_branch_full_pilot
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/branch_train_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/predictive_abstraction
FEATURES="$ROOT/branch_profile_53630/features"
RUN="$ROOT/branch_train_${SLURM_JOB_ID}"
[[ -f "$FEATURES/manifest.json" && ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/configs" "$PROJECT/docs" "$PROJECT/tests" "$RUN/code/"
cp "$PROJECT/scripts/slurm_branch_train.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
sha256sum "$FEATURES/manifest.json" > "$RUN/FEATURE_MANIFEST_SHA256"
export PYTHONPATH="$RUN/code:$ROOT/preflight_53529/deps"
export PA_UPSTREAM_ROOT="$ROOT/preflight_53529/upstream"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -m unittest discover -s "$RUN/code/tests" -v
# Fresh training, not a continuation from the undertrained profile checkpoint.
# Reuse frozen features; no redundant encoding and no test data.
"$PY" -m pa_wm.train_branches --feature-root "$FEATURES" \
  --output "$RUN/output" --seed 20260922
"$PY" -m pa_wm.report_branches --result "$RUN/output/result.json"
