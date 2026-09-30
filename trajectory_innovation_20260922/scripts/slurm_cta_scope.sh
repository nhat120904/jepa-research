#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_scope
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_scope_%A_%a.out
# Usage: this.sh <immutable release> <check|train|compare> [training array id]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?release}; MODE=${2:?mode}; TASK=${SLURM_ARRAY_TASK_ID:-0}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_scope_${MODE}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$TASK"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG=endpoint_trajectory_scope
[[ "$MODE" == train ]] || export WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
case "$MODE" in
  check)
    unit_tests
    SMOKE_RUNS=()
    for SEED in 0 1; do
      for PATH_ARM in 0 1; do
        DEST="$RUN/smoke_s${SEED}_path${PATH_ARM}"
        "$PY" "$CODE/scripts/cta_scope_train.py" --run "$DEST" --features "$ROOT/cta_feat_54490" \
          --seed "$SEED" --path "$PATH_ARM" --device cpu --smoke --steps 2 --decisions 1 --n-train 16 --n-dev 8
        SMOKE_RUNS+=("$DEST")
      done
    done
    "$PY" "$CODE/scripts/cta_scope_compare.py" --runs "${SMOKE_RUNS[@]}" --out "$RUN/smoke_comparison.json" --smoke
    ;;
  train)
    [[ "$TASK" -ge 0 && "$TASK" -le 3 ]] || exit 2
    "$PY" "$CODE/scripts/cta_scope_train.py" --run "$RUN" --features "$ROOT/cta_feat_54490" \
      --seed "$((TASK / 2))" --path "$((TASK % 2))"
    ;;
  compare)
    ARRAY=${3:?training array id}
    "$PY" "$CODE/scripts/cta_scope_compare.py" \
      --runs "$ROOT/cta_scope_train_${ARRAY}_0" "$ROOT/cta_scope_train_${ARRAY}_1" \
             "$ROOT/cta_scope_train_${ARRAY}_2" "$ROOT/cta_scope_train_${ARRAY}_3" \
      --out "$RUN/comparison.json"
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
