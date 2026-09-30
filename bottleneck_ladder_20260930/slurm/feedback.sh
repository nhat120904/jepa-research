#!/usr/bin/env bash
#SBATCH --job-name=feedback
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/bottleneck_ladder/logs/%x_%A_%a.out
# idx 0-1: Cube shards, 2-3: Reacher shards (roots 0-49). RUN_NAME=feedback_v1
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT=/mnt/data/nhatnc129/jepa/bottleneck_ladder/${RUN_NAME:?set RUN_NAME}
mkdir -p "$OUT"
IDX=${SLURM_ARRAY_TASK_ID:?array}
TASKS=(cube cube reacher reacher)
sha256sum /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/*.py > "$OUT/SOURCE_SHA256SUMS_$IDX"
"$PY" /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/feedback_objectives.py \
  --task "${TASKS[$IDX]}" --first 0 --last "${LAST:-50}" --stride 2 --offset $((IDX % 2)) --out-dir "$OUT"
