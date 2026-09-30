#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_reader_refine
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_reader_refine_%A_%a.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}
MODE=${2:?train or closed}
TASK=${SLURM_ARRAY_TASK_ID:-0}
BASE="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_reader_${MODE}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
RUN="$BASE"
if [[ "$MODE" == closed ]]; then RUN="$BASE/shard_$TASK"; fi
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG="reader_refine_$MODE"
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
case "$MODE" in
  train)
    unit_tests
    "$PY" "$CODE/scripts/cta_reader_refine.py" --mode train --run "$RUN" \
      --parent "$ROOT/cta_geometry_e2e_55018/train" --features "$ROOT/cta_geometry_e2e_55018/features" \
      --smoke "$SMOKE" --dev-shard "$ROOT/cta_collect_54489/shard_2000_2049.npz"
    ;;
  closed)
    TRAIN_ID=${3:?training job id}
    [[ "$TASK" -ge 0 && "$TASK" -lt 10 ]] || exit 2
    "$PY" "$CODE/scripts/cta_reader_refine.py" --mode closed --run "$RUN" \
      --train-run "$ROOT/cta_reader_train_$TRAIN_ID" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2100 + TASK * 10))" --count 10
    ;;
  *) exit 2 ;;
esac
