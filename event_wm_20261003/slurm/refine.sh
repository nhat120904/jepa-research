#!/usr/bin/env bash
#SBATCH --job-name=ew_refine
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Self-trained refinement of the event code (pseudo-labels = debounced code away from events).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}
CODE=${CODE:?CODE}; EVENTS=${EVENTS:?EVENTS}
OUT=$RUN_ROOT/refine_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/refine_code.py" --cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --code "$CODE" --events "$EVENTS" ${EXTRA:-} --out "$OUT"
