#!/usr/bin/env bash
#SBATCH --job-name=ew_cubeprep
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Cache visual-cube-triple (first 1500 train episodes + val, with privileged qpos) and the move inventory.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-cube-triple-play-v0
"$OGB_PY" "$PROJECT/scripts/cache_data.py" --data "$DATA" --env "$ENV" --train-episodes 1500 --out "$RUN_ROOT/cache"
"$OGB_PY" "$PROJECT/scripts/cube_inventory.py" --cache "$RUN_ROOT/cache/$ENV" --kind triple --out "$RUN_ROOT/cube_inventory"
