#!/usr/bin/env bash
#SBATCH --job-name=ew_reader
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# CNN code reader on pixels trained on the pipeline's segment-majority pseudo-labels (label-free).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}; EVENTS=${EVENTS:?EVENTS}
OUT=${OUT:-$RUN_ROOT/reader_${SLURM_JOB_ID}_${ENV}}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/train_reader.py" --cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --events "$EVENTS" --out "$OUT" ${EXTRA:-}
