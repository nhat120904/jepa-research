#!/usr/bin/env bash
#SBATCH --job-name=ti_c2_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/c2_eval_%A_%a.out
# Usage: sbatch slurm_c2_eval.sh <c2_train_run_dir>
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
TRAINED=${1:?trained run dir}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
RUN="$ROOT/c2_eval_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
grep -q '"status": "TRAINED"' "$TRAINED/train_report.json" || { echo 'reader not trained' >&2; exit 3; }
[[ ! -e "$CODE" ]] || exit 2
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
START=$((1200 + SLURM_ARRAY_TASK_ID * 50))
"$PY" -m unittest discover -s "$CODE/tests"
"$PY" "$CODE/scripts/c2_eval.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" --trained "$TRAINED" --first "$START" --count 50
