#!/usr/bin/env bash
#SBATCH --job-name=ew_ent
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified front end: SAM 2 entity proposals + video tracking, same rules on every task family; PRIVILEGED diagnostics.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/entities_${SLURM_JOB_ID}
record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/sam2_entities.py" --cache "$RUN_ROOT/cache/visual-cube-triple-play-v0" \
  --project "$RUN_ROOT/cubeA_57120/projection/projection.npz" --out "$OUT" ${ENT_ARGS:-}
for ENV in visual-puzzle-4x5-play-v0 visual-scene-play-v0; do
  "$TORCH_PY" "$PROJECT/scripts/sam2_entities.py" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT" ${ENT_ARGS:-}
done
