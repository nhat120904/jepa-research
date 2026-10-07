#!/usr/bin/env bash
#SBATCH --job-name=ew_scskill
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source_skill_v4"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP="$SCENE_DIR/prep_57771"
OUT="$SCENE_DIR/skill_side_effects_$SLURM_JOB_ID"
mkdir "$OUT"
cd "$SCENE_DIR"
"$PY" "$SCENE_DIR/side_effect_skill_test.py"
find source_skill_v4 -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
"$PY" "$SCENE_DIR/source_skill_v4/u_skill.py" --cache "$PREP/cache/scene-play-v0" --events "$PREP/events" --train-frames 0 --val-frames 0 --steps 10000 --lr 3e-5 --include-side-effects --init-model "$SCENE_DIR/job_57772/skill/u_skill.pt" --out "$OUT/skill" > "$OUT/skill.log" 2>&1
LOOP=("$PY" "$SCENE_DIR/source_skill_v4/u_closed_loop.py" --env scene-v0 --model "$SCENE_DIR/support_57808/u_model.pt" --state-layout "$PREP/front/layout.json" --events "$PREP/events" --cache "$PREP/cache/scene-play-v0" --workers 5 --max-expansions 20000)
# Diagnostic on the inspected cases, then a predeclared fresh seed for both frozen-control and new-skill arms.
"${LOOP[@]}" --skill "$OUT/skill/u_skill.pt" --tasks 4 5 --episodes 20 --seed 4 --out "$OUT/new_skill_dev_seed4"
"${LOOP[@]}" --skill "$SCENE_DIR/job_57772/skill/u_skill.pt" --episodes 20 --seed 5 --out "$OUT/old_skill_seed5"
"${LOOP[@]}" --skill "$OUT/skill/u_skill.pt" --episodes 20 --seed 5 --out "$OUT/new_skill_seed5"
