#!/usr/bin/env bash
#SBATCH --job-name=ew_sccmp
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source_boundary_v3"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP="$SCENE_DIR/prep_57771"
OUT="$SCENE_DIR/compare_$SLURM_JOB_ID"
mkdir "$OUT"
"$PY" "$SCENE_DIR/boundary_test.py"
cd "$SCENE_DIR"
find source_boundary_v3 -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
LOOP=("$PY" "$SCENE_DIR/source_boundary_v3/u_closed_loop.py" --env scene-v0 --state-layout "$PREP/front/layout.json" --skill "$SCENE_DIR/job_57772/skill/u_skill.pt" --events "$PREP/events" --cache "$PREP/cache/scene-play-v0" --workers 4 --max-expansions 20000 --device cpu)
"${LOOP[@]}" --model "$SCENE_DIR/support_57808/u_model.pt" --episodes 20 --seed 3 --out "$OUT/support_seed3"
"${LOOP[@]}" --model "$SCENE_DIR/job_57772/wm/u_model.pt" --episodes 20 --seed 4 --out "$OUT/base_seed4"
"${LOOP[@]}" --model "$SCENE_DIR/support_57808/u_model.pt" --episodes 20 --tasks 4 5 --seed 4 --acted-rest-only --out "$OUT/acted_rest_seed4"
