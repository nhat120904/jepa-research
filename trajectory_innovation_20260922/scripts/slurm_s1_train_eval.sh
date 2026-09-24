#!/usr/bin/env bash
#SBATCH --job-name=ti_s1_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/s1_train_%j.out
# Usage: sbatch slurm_s1_train_eval.sh <s1_collect_run_dir>
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
COLLECT=${1:?collect run dir}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
C2="$ROOT/c2_train_53938"
RUN="$ROOT/s1_train_${SLURM_JOB_ID}"
[[ ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$PROJECT/docs" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS"
export PYTHONPATH="$RUN/code:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
"$PY" -m unittest discover -s "$RUN/code/tests"
"$PY" "$RUN/code/scripts/s1_train_eval.py" --run "$RUN" --collect "$COLLECT" --prep "$PREP" --smoke "$SMOKE" --c2 "$C2"
