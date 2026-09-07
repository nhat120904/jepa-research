#!/usr/bin/env bash
#SBATCH --job-name=spwm_final
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_final_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
SPLIT=${SPLIT:?submit with SPLIT=train|val}
OUT_TAG=${OUT_TAG:-$SPLIT}
case "$SPLIT" in
  train) SRC="$DATA_ROOT/visual-scene-play-v0.npz" ;;
  val)   SRC="$DATA_ROOT/visual-scene-play-v0-val.npz" ;;
  *) echo "unknown SPLIT=$SPLIT" >&2; exit 2 ;;
esac
sha256sum "$PROJECT/scripts/build_scene_cache.py" "$PROJECT/scripts/slurm_cache_finalize.sh"
"$PY" "$PROJECT/scripts/build_scene_cache.py" --stage finalize \
  --dataset "$SRC" --out-dir "$CACHE_ROOT/cache/$OUT_TAG"
echo "DONE $(date -u +%FT%TZ)"
