#!/usr/bin/env bash
#SBATCH --job-name=ti_gate_abc
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/gate_abc_%A_%a.out
# Usage: sbatch slurm_gate_abc.sh <smoke_run_dir> [first_root=1000] [roots_per_shard=25]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
SMOKE=${1:?smoke run dir}
FIRST=${2:-1000}
PER=${3:-25}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
RUN="$ROOT/gate_abc_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ -f "$SMOKE/smoke.json" && ! -e "$CODE" ]] || exit 2
grep -q '"status": "SMOKE_PASS"' "$SMOKE/smoke.json" || { echo 'smoke did not pass' >&2; exit 3; }
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
START=$((FIRST + SLURM_ARRAY_TASK_ID * PER))
"$PY" -m unittest discover -s "$CODE/tests"
"$PY" "$CODE/scripts/gate_abc.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" --first "$START" --count "$PER"
