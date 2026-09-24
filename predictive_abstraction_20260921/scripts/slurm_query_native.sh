#!/usr/bin/env bash
#SBATCH --job-name=pa_final_query_native
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/query_native_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/predictive_abstraction
FEATURES="$ROOT/branch_profile_53630/features"
CHECKPOINT="$ROOT/fixed_summary_53698/output/checkpoints.pt"
RUN="$ROOT/query_native_${SLURM_JOB_ID}"
[[ -f "$FEATURES/manifest.json" && -f "$CHECKPOINT" && ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/docs" "$PROJECT/tests" "$RUN/code/"
cp "$PROJECT/scripts/slurm_query_native.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
sha256sum "$FEATURES/manifest.json" "$CHECKPOINT" > "$RUN/INPUT_SHA256SUMS"
export PYTHONPATH="$RUN/code:$ROOT/preflight_53529/deps"
export PA_UPSTREAM_ROOT="$ROOT/preflight_53529/upstream"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -m unittest discover -s "$RUN/code/tests" -v
"$PY" -m pa_wm.query_native --features "$FEATURES" --checkpoint "$CHECKPOINT" --output "$RUN/output"
