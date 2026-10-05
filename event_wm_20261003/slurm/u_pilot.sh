#!/usr/bin/env bash
#SBATCH --job-name=ew_upilot
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified front end v2 pilot: sparse SAM 2 frames + identity anchoring -> u_events, on every task family.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/upilot_${SLURM_JOB_ID}
record_source "$OUT/src"
for ENV in ${ENVS:-visual-cube-triple-play-v0 visual-puzzle-4x5-play-v0 visual-scene-play-v0}; do
  "$TORCH_PY" "$PROJECT/scripts/sam2_frames.py" --cache "$RUN_ROOT/cache/$ENV" --train-episodes "${TEP:-10}" --val-episodes "${VEP:-5}" \
    --out "$OUT/$ENV/front" ${FRONT_ARGS:-}
  "$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$OUT/$ENV/front" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT/$ENV/events"
done
