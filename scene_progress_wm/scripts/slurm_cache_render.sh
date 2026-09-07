#!/usr/bin/env bash
#SBATCH --job-name=spwm_render
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_render_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SPLIT=${SPLIT:?submit with SPLIT=train|val}
OUT_TAG=${OUT_TAG:-$SPLIT}
NUM_SHARDS=${NUM_SHARDS:?submit with NUM_SHARDS}
QUALITY=${QUALITY:-fast}
STRIDE=${STRIDE:-5}
SHARD=${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array}
case "$SPLIT" in
  train) SRC="$DATA_ROOT/visual-scene-play-v0.npz" ;;
  val)   SRC="$DATA_ROOT/visual-scene-play-v0-val.npz" ;;
  *) echo "unknown SPLIT=$SPLIT" >&2; exit 2 ;;
esac
sha256sum "$PROJECT/scripts/build_scene_cache.py" "$PROJECT/scripts/slurm_cache_render.sh"
"$PY" "$PROJECT/scripts/build_scene_cache.py" --stage render \
  --dataset "$SRC" --out-dir "$CACHE_ROOT/cache/$OUT_TAG" \
  --render-size 64 --camera front_pixels \
  --pixel-stride "$STRIDE" --render-quality "$QUALITY" \
  --shard-index "$SHARD" --num-shards "$NUM_SHARDS"
echo "DONE $(date -u +%FT%TZ)"
