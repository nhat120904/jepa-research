#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_onpol8
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_onpol8_%x_%A_%a.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=${1:?immutable source release}
MODE=${2:?smoke/collect/encode/train/closed/aggregate}
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE="$TI/cta_onpolicy8_${MODE}_${JOB}"
if [[ "$MODE" == collect || "$MODE" == closed ]]; then
  RUN="$BASE/shard_$TASK"
else
  RUN="$BASE"
fi
CODE="$RUN/code"
HASHES="$RUN/CODE_SHA256SUMS"
[[ ! -e "$CODE" ]] || { echo "Existing code snapshot: $CODE" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG=cta_onpolicy8
source "$PROJECT/scripts/cta_env.sh"
snapshot "$HASHES"
S="$CODE/scripts"
PARENT="$ROOT/cta_geometry_e2e_55018/train"
R4="$ROOT/cta_plus_train_55149/train"
R3="$ROOT/cta_parallel_train_55077"
OLD="$ROOT/cta_geometry_e2e_55018/features"
ARMS=P0,GEOM8,CTA4,CTA8O,DIRECT8O,ENDPOINT8O
LOG=CTA4,DIRECT4,CTA8O,DIRECT8O,ENDPOINT8O
case "$MODE" in
  smoke)
    unit_tests
    mkdir -p "$RUN/smoke/collection"
    "$PY" "$S/cta_collect_onpolicy8.py" --out "$RUN/smoke/collection" --prep "$PREP" --smoke "$SMOKE" \
      --parent "$PARENT" --r4 "$R4" --first 39994 --count 2 --max-decisions 2
    "$PY" "$S/cta_collect_onpolicy8.py" --out "$RUN/smoke/collection" --prep "$PREP" --smoke "$SMOKE" \
      --parent "$PARENT" --r4 "$R4" --first 39996 --count 2 --max-decisions 2
    "$PY" "$S/cta_encode_onpolicy8.py" --collect "$RUN/smoke/collection" --old-features "$OLD" \
      --out "$RUN/smoke/features" --train-first 39994 --train-last 39995 --dev-first 39996 --dev-last 39997
    "$PY" "$S/cta_train_onpolicy8.py" --out "$RUN/smoke/train" --features "$RUN/smoke/features" \
      --parent "$PARENT" --r4 "$R4" --r3 "$R3" --steps 2 --eval-every 1 --batch 4 --micro 2 --limit 8
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN/smoke/closed/shard_0" \
      --parent "$PARENT" --r4 "$R4" --onpolicy8 "$RUN/smoke/train" --prep "$PREP" --smoke "$SMOKE" \
      --first 2200 --count 2 --max-decisions 2 --arms "$ARMS" --log-scorers "$LOG"
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN/smoke/aggregate" \
      --closed-run "$RUN/smoke/closed" --pairs CTA8O-DIRECT8O,CTA8O-ENDPOINT8O,CTA8O-CTA4
    echo ONPOLICY8_SMOKE_OK
    ;;
  collect)
    [[ "$TASK" -ge 0 && "$TASK" -le 8 ]] || exit 2
    "$PY" "$S/cta_collect_onpolicy8.py" --out "$BASE" --prep "$PREP" --smoke "$SMOKE" \
      --parent "$PARENT" --r4 "$R4" --first "$((32250 + TASK * 50))" --count 50
    ;;
  encode)
    "$PY" "$S/cta_encode_onpolicy8.py" --collect "$TI/cta_onpolicy8_collect_${3:?collect id}" \
      --old-features "$OLD" --out "$RUN/features" --train-first 32250 --train-last 32649 \
      --dev-first 32650 --dev-last 32699
    ;;
  train)
    "$PY" "$S/cta_train_onpolicy8.py" --out "$RUN/train" --parent "$PARENT" --r4 "$R4" --r3 "$R3" \
      --features "$TI/cta_onpolicy8_encode_${3:?encode id}/features"
    ;;
  closed)
    [[ "$TASK" -ge 0 && "$TASK" -le 7 ]] || exit 2
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r4 "$R4" \
      --onpolicy8 "$TI/cta_onpolicy8_train_${3:?train id}/train" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2200 + TASK * 25))" --count 25 --arms "$ARMS" --log-scorers "$LOG"
    ;;
  aggregate)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" \
      --closed-run "$TI/cta_onpolicy8_closed_${3:?closed id}" \
      --pairs CTA8O-DIRECT8O,CTA8O-ENDPOINT8O,CTA8O-CTA4,CTA8O-GEOM8 --expect-roots 2200 2399
    ;;
  *) echo "Unknown mode $MODE" >&2; exit 2 ;;
esac
