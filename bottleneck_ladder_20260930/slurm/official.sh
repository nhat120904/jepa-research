#!/usr/bin/env bash
#SBATCH --job-name=official
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/bottleneck_ladder/logs/%x_%j.out
# Upstream Reacher evaluator, history_len 1 vs 3, seeds 42-44 (6 runs sequentially).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT=/mnt/data/nhatnc129/jepa/bottleneck_ladder/official_${SLURM_JOB_ID}
mkdir -p "$OUT"
sha256sum /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/official_eval.py > "$OUT/SOURCE_SHA256SUMS"
for SEED in 42 43 44; do for H in 1 3; do
  "$PY" /home/nhatnc129/nhat.nc/jepa-research/bottleneck_ladder_20260930/scripts/official_eval.py \
    --task reacher --history $H --seed $SEED --out "$OUT/reacher_h${H}_s${SEED}.json"
done; done
