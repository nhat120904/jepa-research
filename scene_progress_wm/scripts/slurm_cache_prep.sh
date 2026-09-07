#!/usr/bin/env bash
#SBATCH --job-name=spwm_prep
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_prep_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SPLIT=${SPLIT:?submit with SPLIT=train|val}
OUT_TAG=${OUT_TAG:-$SPLIT}
QUALITY=${QUALITY:-fast}
STRIDE=${STRIDE:-5}
case "$SPLIT" in
  train) SRC="$DATA_ROOT/visual-scene-play-v0.npz" ;;
  val)   SRC="$DATA_ROOT/visual-scene-play-v0-val.npz" ;;
  *) echo "unknown SPLIT=$SPLIT" >&2; exit 2 ;;
esac
sha256sum "$PROJECT/scripts/build_scene_cache.py" "$PROJECT/scripts/slurm_cache_prep.sh"
df -h /mnt/data | tail -1
"$PY" "$PROJECT/scripts/build_scene_cache.py" --stage prep \
  --dataset "$SRC" --out-dir "$CACHE_ROOT/cache/$OUT_TAG" \
  --render-size 64 --camera front_pixels \
  --pixel-stride "$STRIDE" --render-quality "$QUALITY"
echo "DONE $(date -u +%FT%TZ)"
