#!/usr/bin/env bash
#SBATCH --job-name=ew_cache
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Uncompressed, memory-mappable copies of the first TEP train episodes (+ all val) per puzzle size.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
for ENV in ${ENVS:-visual-puzzle-4x5-play-v0 visual-puzzle-4x6-play-v0 visual-puzzle-3x3-play-v0 visual-puzzle-4x4-play-v0}; do
  "$OGB_PY" "$PROJECT/scripts/cache_data.py" --data "$DATA" --env "$ENV" --train-episodes "${TEP:-1500}" --out "$RUN_ROOT/cache"
done
