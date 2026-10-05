#!/usr/bin/env bash
#SBATCH --job-name=ew_sam2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Front-end fallback probe: SAM 2.1 point-grid segmentation on puzzle / cube / scene frames.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/sam2_${SLURM_JOB_ID}
record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/sam2_probe.py" --cache "$RUN_ROOT/cache/visual-cube-triple-play-v0" \
  --project "$RUN_ROOT/cubeA_57120/projection/projection.npz" --out "$OUT"
for ENV in visual-puzzle-4x5-play-v0 visual-scene-play-v0; do
  "$TORCH_PY" "$PROJECT/scripts/sam2_probe.py" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT"
done
