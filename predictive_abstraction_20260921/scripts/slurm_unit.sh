#!/usr/bin/env bash
#SBATCH --job-name=pa_unit
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/unit_%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
UPSTREAM=/mnt/data/nhatnc129/jepa/dino_wm_original
RUN=/mnt/data/nhatnc129/jepa/predictive_abstraction/unit_${SLURM_JOB_ID}
[[ ! -e "$RUN" ]] || { echo "Refusing existing run" >&2; exit 2; }
mkdir -p "$RUN/code" "$RUN/upstream/env" "$RUN/deps"
cp -r "$PROJECT/pa_wm" "$PROJECT/tests" "$RUN/code/"
cp -r "$UPSTREAM/env/wall" "$RUN/upstream/env/"
/home/nhatnc129/.local/bin/uv pip install --python "$PY" --no-deps --target "$RUN/deps" \
  gym==0.26.2 gym_notices==0.0.8
export PA_UPSTREAM_ROOT="$RUN/upstream"
export PYTHONPATH="$RUN/code:$RUN/deps" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$RUN/code"
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$RUN/tests.log"
