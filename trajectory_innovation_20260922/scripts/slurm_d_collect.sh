#!/usr/bin/env bash
#SBATCH --job-name=ti_d_collect
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/d_collect_%A_%a.out
# Gate D collection. Tasks 0-7: training roots 30250-30649; tasks 8-9: held-out roots 2000-2099.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
RUN="$ROOT/d_collect_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CODE" ]] || exit 2
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
T=$SLURM_ARRAY_TASK_ID
if (( T < 8 )); then START=$((30250 + T * 50)); else START=$((2000 + (T - 8) * 50)); fi
"$PY" -m unittest discover -s "$CODE/tests"
"$PY" "$CODE/scripts/d_collect.py" --out "$RUN" --prep "$PREP" --smoke "$SMOKE" --first "$START" --count 50
