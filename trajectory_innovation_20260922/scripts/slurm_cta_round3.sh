#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_parallel_r3
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_parallel_r3_%A_%a.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}
MODE=${2:?train or closed}
TASK=${SLURM_ARRAY_TASK_ID:-0}
BASE="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_parallel_${MODE}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
RUN="$BASE"
if [[ "$MODE" == closed ]]; then RUN="$BASE/shard_$TASK"; fi
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG="parallel_r3_$MODE"
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
case "$MODE" in
  train)
    unit_tests
    RESUME_ARGS=()
    if [[ -n "${3:-}" ]]; then RESUME_ARGS=(--resume-from "$3"); fi
    # Forward Slurm's early time-limit warning to Python so it checkpoints
    # after the next full update. Periodic checkpoints also protect evaluations.
    trap 'kill -USR1 "$ROUND3_PID" 2>/dev/null || true' USR1
    "$PY" "$CODE/scripts/cta_round3.py" --mode train --run "$RUN" \
      --parent "$ROOT/cta_geometry_e2e_55018/train" --features "$ROOT/cta_geometry_e2e_55018/features" \
      --collection "$ROOT/cta_collect_54489" --smoke "$SMOKE" "${RESUME_ARGS[@]}" &
    ROUND3_PID=$!
    set +e
    wait "$ROUND3_PID"
    ROUND3_RC=$?
    if kill -0 "$ROUND3_PID" 2>/dev/null; then
      wait "$ROUND3_PID"
      ROUND3_RC=$?
    fi
    exit "$ROUND3_RC"
    ;;
  closed)
    TRAIN_ID=${3:?training job id}
    [[ "$TASK" -ge 0 && "$TASK" -lt 10 ]] || exit 2
    "$PY" "$CODE/scripts/cta_round3.py" --mode closed --run "$RUN" \
      --train-run "$ROOT/cta_parallel_train_$TRAIN_ID" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2100 + TASK * 10))" --count 10
    ;;
  *) exit 2 ;;
esac
