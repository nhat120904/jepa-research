#!/usr/bin/env bash
#SBATCH --job-name=pa_wm_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/train_%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
MODE="${MODE:-profile}"
[[ "$MODE" == "profile" || "$MODE" == "full" ]] || { echo "Bad MODE" >&2; exit 2; }
ENCODE_RUN="${ENCODE_RUN:?Set ENCODE_RUN to a completed encode job id}"
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
FEATURE_ROOT=/mnt/data/nhatnc129/jepa/predictive_abstraction/encode_${ENCODE_RUN}/output
[[ -f "$FEATURE_ROOT/manifest.json" ]] || { echo "Missing completed feature manifest" >&2; exit 3; }
RUN=/mnt/data/nhatnc129/jepa/predictive_abstraction/train_${MODE}_${SLURM_JOB_ID}
[[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 4; }
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/configs" "$PROJECT/docs" "$RUN/code/"
cp "$PROJECT/scripts/slurm_train.sh" "$RUN/code/"
sed "s@ENCODE_RUN@encode_${ENCODE_RUN}@g" "$RUN/code/configs/train_pilot.json" > "$RUN/code/configs/train_resolved.json"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
export PYTHONPATH="$RUN/code" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
EXTRA=()
[[ "$MODE" == "profile" ]] && EXTRA+=(--profile)
"$PY" -m pa_wm.train --config "$RUN/code/configs/train_resolved.json" \
  --run-dir "$RUN/output" "${EXTRA[@]}"
