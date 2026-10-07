#!/usr/bin/env bash
#SBATCH --job-name=ew_scsup
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source_support_v2"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
OUT="$SCENE_DIR/support_$SLURM_JOB_ID"
"$PY" "$SCENE_DIR/support_test.py"
"$PY" "$SCENE_DIR/train_support.py" --root "$SCENE_DIR" --out "$OUT"
cd "$SCENE_DIR"
find source_support_v2 -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
PREP="$SCENE_DIR/prep_57771"
LOOP=("$PY" "$SCENE_DIR/source_support_v2/u_closed_loop.py" --env scene-v0 --model "$OUT/u_model.pt" --state-layout "$PREP/front/layout.json" --skill "$SCENE_DIR/job_57772/skill/u_skill.pt" --events "$PREP/events" --cache "$PREP/cache/scene-play-v0" --workers 4 --max-expansions 20000 --device cpu)
"${LOOP[@]}" --episodes 1 --seed 0 --out "$OUT/loop_dev"
"${LOOP[@]}" --episodes 20 --seed 4 --out "$OUT/loop_seed4"
