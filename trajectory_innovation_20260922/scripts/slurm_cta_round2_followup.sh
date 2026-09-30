#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_r2_followup
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_r2_followup_%A_%a.out
# Usage: sbatch [resource overrides] this.sh <immutable release> <audit|preflight|closed|aggregate|wm_audit> [closed array id]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}; MODE=${2:?mode}
TASK=${SLURM_ARRAY_TASK_ID:-0}
BASE="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_${MODE}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
RUN="$BASE"; CODE="$BASE/code_$TASK"
[[ ! -e "$CODE" ]] || exit 2
export CTA_TAG="r2_$MODE"
[[ "$MODE" != audit && "$MODE" != aggregate ]] || export WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$BASE/CODE_SHA256SUMS_$TASK"
case "$MODE" in
  audit)
    unit_tests
    "$PY" "$CODE/scripts/cta_round2_followup.py" --mode audit --features "$ROOT/cta_feat_54490" \
      --train-runs "$ROOT/cta_train_54717" "$ROOT/cta_r2_54933_0" "$ROOT/cta_r2_54933_1" --out "$RUN/audit.json"
    ;;
  preflight|closed)
    if [[ "$MODE" == preflight ]]; then
      ARM=$TASK; START=2100; COUNT=0
    else
      ARM=$((TASK / 10)); START=$((2100 + TASK % 10 * 10)); COUNT=10
    fi
    [[ "$ARM" == 0 || "$ARM" == 1 ]] || exit 2
    NAMES=(control method)
    RUN="$BASE/${NAMES[$ARM]}"
    mkdir -p "$RUN"
    export WANDB_DIR="$RUN"
    # Reuse the existing inference/preflight contract without changing any model.
    "$PY" "$CODE/scripts/cta_closed_loop.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" \
      --train-run "$ROOT/cta_r2_54933_$ARM" --dev-shard "$ROOT/cta_collect_54489/shard_2000_2049.npz" \
      --first "$START" --count "$COUNT" --arms P0,CODE8,CTA8,DIRECT8,CTA8E
    ;;
  aggregate)
    CLOSED=${3:?closed loop array id}
    "$PY" "$CODE/scripts/cta_round2_followup.py" --mode aggregate --closed-run "$ROOT/cta_r2_closed_$CLOSED" \
      --out "$RUN/comparison.json"
    ;;
  wm_audit)
    "$PY" "$CODE/scripts/cta_wm_evidence_audit.py" --train-run "$ROOT/cta_r2_54933_0" --run "$RUN"
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
