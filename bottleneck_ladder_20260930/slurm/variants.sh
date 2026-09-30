#!/usr/bin/env bash
#SBATCH --job-name=ctxvar
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/bottleneck_ladder/logs/%x_%A_%a.out
# Reacher context controls; array index = shard. RUN_NAME=variants_v1 sbatch --array=0-3%2 slurm/variants.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT=/mnt/data/nhatnc129/jepa/bottleneck_ladder/${RUN_NAME:?set RUN_NAME}
mkdir -p "$OUT"
sha256sum /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/*.py > "$OUT/SOURCE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
"$PY" /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/context_variants.py \
  --stride 4 --offset "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT" ${EXTRA_ARGS:-}
