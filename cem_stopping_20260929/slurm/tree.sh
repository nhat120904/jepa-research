#!/usr/bin/env bash
#SBATCH --job-name=cemstop_tree
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%A_%a.out
# Stopping trees for development roots. Array index -> (task, shard).
# Usage: RUN_NAME=dev_v1 SHARDS=4 FIRST=0 LAST=100 sbatch --array=0-7%4 slurm/tree.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
SHARDS=${SHARDS:-4}
FIRST=${FIRST:-0}
LAST=${LAST:-100}
TASK_NAMES=(cube reacher)
IDX=${SLURM_ARRAY_TASK_ID:?submit as an array}
TASK=${TASK_NAMES[$((IDX / SHARDS))]}
SHARD=$((IDX % SHARDS))
RUN_DIR="$RUN_ROOT/${RUN_NAME:?set RUN_NAME}"
record_source "$RUN_DIR/source_${SLURM_ARRAY_JOB_ID}_${IDX}"
"$PY" "$PROJECT/scripts/run_tree.py" --task "$TASK" --first "$FIRST" --last "$LAST" \
  --stride "$SHARDS" --offset "$SHARD" --label-roots "${LABEL_ROOTS:-20}" \
  --out-dir "$RUN_DIR" ${EXTRA_ARGS:-}
