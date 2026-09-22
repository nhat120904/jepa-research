#!/usr/bin/env bash
#SBATCH --job-name=pa_metric_bridge
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/metric_bridge_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/predictive_abstraction
FEATURES="$ROOT/branch_profile_53630/features"
METRIC="$ROOT/codec_rescue_53659/output/diagnostic_checkpoints.pt"
RUN="$ROOT/metric_bridge_${SLURM_JOB_ID}"
[[ -f "$FEATURES/manifest.json" && -f "$METRIC" && ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/docs" "$PROJECT/tests" "$RUN/code/"
cp "$PROJECT/scripts/slurm_metric_bridge.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
sha256sum "$FEATURES/manifest.json" "$METRIC" > "$RUN/INPUT_SHA256SUMS"
export PYTHONPATH="$RUN/code:$ROOT/preflight_53529/deps"
export PA_UPSTREAM_ROOT="$ROOT/preflight_53529/upstream"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -m unittest discover -s "$RUN/code/tests" -v
"$PY" -m pa_wm.metric_bridge --features "$FEATURES" --metric-checkpoint "$METRIC" --output "$RUN/output"
