#!/usr/bin/env bash
#SBATCH --job-name=pa_rgb_preflight
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/preflight_%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
UPSTREAM=/mnt/data/nhatnc129/jepa/dino_wm_original
RUN=/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_${SLURM_JOB_ID}
CONFIG_NAME="${CONFIG_NAME:-preflight.json}"
MODE="${MODE:-full}"
[[ "$MODE" == "full" || "$MODE" == "screen_only" ]] || { echo "Bad MODE" >&2; exit 3; }
[[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 2; }
mkdir -p "$RUN/code" "$RUN/upstream/env" "$RUN/deps"
cp -r "$PROJECT/pa_wm" "$PROJECT/tests" "$PROJECT/configs" "$PROJECT/docs" "$RUN/code/"
cp "$PROJECT/scripts/slurm_preflight.sh" "$RUN/code/"
cp -r "$UPSTREAM/env/wall" "$RUN/upstream/env/"
cp "$UPSTREAM/LICENSE" "$RUN/upstream/"
find "$RUN/code" "$RUN/upstream" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
git -C /home/nhatnc129/nhat.nc/jepa-research rev-parse HEAD > "$RUN/repository_head.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
# Legacy gym is required solely as the upstream Env base class. Install into this
# run's private target; never modify the shared policy environment.
/home/nhatnc129/.local/bin/uv pip install --python "$PY" --no-deps --target "$RUN/deps" \
    gym==0.26.2 gym_notices==0.0.8
export PA_UPSTREAM_ROOT="$RUN/upstream"
export PYTHONPATH="$RUN/code:$RUN/deps"
cd "$RUN/code"
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$RUN/tests.log"
EXTRA=()
[[ "$MODE" == "screen_only" ]] && EXTRA+=(--screen-only)
"$PY" -m pa_wm.preflight --config "configs/$CONFIG_NAME" --run-dir "$RUN" \
    --upstream-root "$RUN/upstream" "${EXTRA[@]}"
