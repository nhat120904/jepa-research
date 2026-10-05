#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_hit
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_hit_%x_%A_%a.out
# CPU defaults. GPU modes require explicit --partition=mig --gres=... overrides.
# Submit this script from an immutable source release after quota/duplicate checks.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=${CTA_SOURCE_ROOT:?Set CTA_SOURCE_ROOT to the immutable release/code directory with sbatch --export}
MODE=${1:?mode: tests, smoke, train, closed, aggregate}
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE=${BASE_CHECKPOINT:-$TI/cta_replan15_train_56504/train/cta_v2.pt}
ARMS=${ARMS:-P0,GEOM8,CTA_BASE,END_BASE,DIR_BASE,CTA_CTRL,END_CTRL,DIR_CTRL,CTA_HIT,END_HIT,DIR_HIT}
ARMS=${ARMS//:/,} # Colon-separated overrides work with comma-delimited sbatch --export.
FIRST=${FIRST:-2200}
COUNT=${COUNT:-10}
case "$MODE" in
  tests|aggregate) ;;
  smoke|train|closed)
    [[ -n "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}" ]] || {
      echo 'GPU mode requires an explicit GPU sbatch allocation' >&2; exit 2;
    } ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
if [[ "$MODE" == closed || "$MODE" == aggregate ]]; then
  [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo 'Expected upstream job ID' >&2; exit 2; }
fi
if [[ "$MODE" == closed ]]; then
  RUN="$TI/cta_hit_closed_${JOB}/shard_${TASK}"
else
  RUN="$TI/cta_hit_${MODE}_${JOB}"
fi
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || { echo "Refusing to overwrite $RUN" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG=hit15
source "$PROJECT/scripts/cta_env.sh"
mkdir -p "$RUN"
snapshot "$RUN/CODE_SHA256SUMS"
S="$CODE/scripts"
printf 'source=%s mode=%s job=%s task=%s\n' "$PROJECT" "$MODE" "$JOB" "$TASK" > "$RUN/launch.txt"
closed() {
  "$PY" "$S/cta_hit_closed.py" --run "$1" --checkpoint "BASE=$BASE" \
    --checkpoint "CTRL=$2/control/cta_v2.pt" --checkpoint "HIT=$2/hit/cta_v2.pt" \
    --arms "$ARMS" --first "$3" --count "$4" "${@:5}"
}
case "$MODE" in
  tests)
    unit_tests
    echo CTA_HIT_CPU_TESTS_OK
    ;;
  smoke)
    unit_tests
    "$PY" "$S/cta_hit_train.py" --base "$BASE" --out "$RUN/train" \
      --steps 20 --warmup 2 --decisions 2 --eval-every 10 --log-every 10 \
      --train-limit 64 --selection-limit 8
    closed "$RUN/closed" "$RUN/train" 2200 2 --max-decisions 2 --qualify-decisions 1
    echo CTA_HIT_SMOKE_OK
    ;;
  train)
    "$PY" "$S/cta_hit_train.py" --base "$BASE" --out "$RUN/train" \
      --steps "${STEPS:-2400}" --eval-every "${EVAL_EVERY:-1200}" \
      --decisions "${DECISIONS:-16}" --seed "${SEED:-0}"
    ;;
  closed)
    TRAIN="$TI/cta_hit_train_${2}/train"
    closed "$RUN/results" "$TRAIN" "$((FIRST + TASK * COUNT))" "$COUNT"
    ;;
  aggregate)
    CLOSED="$TI/cta_hit_closed_${2}"
    # Shard reports live in results/ beside immutable code and launch metadata.
    shopt -s nullglob
    INPUTS=("$CLOSED"/shard_*/results)
    [[ ${#INPUTS[@]} -gt 0 ]] || { echo "No closed-loop outputs in $CLOSED" >&2; exit 2; }
    "$PY" "$S/cta_hit_aggregate.py" --closed-run "${INPUTS[@]}" --out "$RUN/aggregate" \
      --expect-roots "${EXPECTED_FIRST:-2200}" "${EXPECTED_LAST:-2219}"
    ;;
esac
echo "DONE $MODE $(date -u +%FT%TZ)"
