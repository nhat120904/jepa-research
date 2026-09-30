#!/usr/bin/env bash
#SBATCH --job-name=consensus
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/consensus_planning/logs/%x_%A_%a.out
# Array index -> (task, shard). Usage: RUN_NAME=dev_v1 SHARDS=4 sbatch --array=0-7%2 slurm/run.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
CPROJ=/home/nhatnc129/nhat.nc/jepa-research/consensus_planning_20260929
ROOT=/mnt/data/nhatnc129/jepa/consensus_planning
SHARDS=${SHARDS:-4}
TASK_NAMES=(cube reacher)
IDX=${SLURM_ARRAY_TASK_ID:?array}
TASK=${TASK_NAMES[$((IDX / SHARDS))]}
OUT="$ROOT/${RUN_NAME:?set RUN_NAME}"
mkdir -p "$OUT/source_${SLURM_ARRAY_JOB_ID}_${IDX}"
(cd /home/nhatnc129/nhat.nc/jepa-research && sha256sum consensus_planning_20260929/scripts/*.py cem_stopping_20260929/cemstop/*.py) > "$OUT/source_${SLURM_ARRAY_JOB_ID}_${IDX}/SOURCE_SHA256SUMS"
"$PY" "$CPROJ/scripts/run_consensus.py" --task "$TASK" --first "${FIRST:-0}" --last "${LAST:-100}" \
  --stride "$SHARDS" --offset $((IDX % SHARDS)) --out-dir "$OUT"
