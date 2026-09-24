#!/usr/bin/env bash
#SBATCH --job-name=ti_c2_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/c2_train_%j.out
# Usage: sbatch slurm_c2_train.sh [--seed N] [--no-control]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
RUN="$ROOT/c2_train_${SLURM_JOB_ID}"
[[ -f "$SMOKE/smoke.json" && ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS"
export PYTHONPATH="$RUN/code:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
"$PY" -m unittest discover -s "$RUN/code/tests" -v
"$PY" "$RUN/code/scripts/c2_train.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" "$@"
