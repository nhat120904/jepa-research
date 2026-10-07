#!/usr/bin/env bash
#SBATCH --job-name=ew_scprep
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
OUT="$SCENE_DIR/prep_$SLURM_JOB_ID"
mkdir "$OUT"
cd "$SCENE_DIR"
find source -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
"$PY" "$SCENE_DIR/test_scene_state.py"
"$PY" "$SCENE_DIR/source/s_entities.py" --data /mnt/data/nhatnc129/jepa/ogbench/data --env scene-play-v0 --train-episodes 3000 --val-episodes 100 --out "$OUT"
"$PY" "$SCENE_DIR/source/u_events.py" --entities "$OUT/front" --cache "$OUT/cache/scene-play-v0" --thresholds-from "$OUT/thr" --per-frame --out "$OUT/events"
"$PY" "$SCENE_DIR/verify_prepared.py" --out "$OUT"
