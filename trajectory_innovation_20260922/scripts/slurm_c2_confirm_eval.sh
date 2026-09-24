#!/usr/bin/env bash
#SBATCH --job-name=ti_c2c_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/c2c_eval_%A_%a.out
# Usage: sbatch slurm_c2_confirm_eval.sh <r0_train_run_dir> <r1_train_run_dir>
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
R0=${1:?r0 run dir}; R1=${2:?r1 run dir}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
RUN="$ROOT/c2_confirm_eval_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
for T in "$R0" "$R1"; do grep -q '"status": "TRAINED"' "$T/train_report.json" || { echo "reader not trained: $T" >&2; exit 3; }; done
[[ ! -e "$CODE" ]] || exit 2
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
START=$((1300 + SLURM_ARRAY_TASK_ID * 50))
"$PY" -m unittest discover -s "$CODE/tests"
"$PY" "$CODE/scripts/c2_confirm_eval.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" --r0 "$R0" --r1 "$R1" --first "$START" --count 50
