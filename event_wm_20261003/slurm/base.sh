#!/usr/bin/env bash
#SBATCH --job-name=ew_base
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# LeWM-recipe base world model (frozen encoder source) for one puzzle size, from the event_wm cache.
#   ENV=visual-puzzle-4x6-play-v0 sbatch slurm/base.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:?ENV}
OUT=$RUN_ROOT/base_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
"$TORCH_PY" "$CG/scripts/train_wm.py" --data "$DATA" --env "$ENV" --cache "$RUN_ROOT/cache" \
  --cache-frames "${FRAMES:-1000000}" --max-steps "${STEPS:-60000}" --out "$OUT"
