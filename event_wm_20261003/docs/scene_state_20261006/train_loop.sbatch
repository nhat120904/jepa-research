#!/usr/bin/env bash
#SBATCH --job-name=ew_scene
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP="$SCENE_DIR/prep_57771"
OUT="$SCENE_DIR/job_$SLURM_JOB_ID"
mkdir "$OUT"
cd "$SCENE_DIR"
find source -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
"$PY" "$SCENE_DIR/test_scene_state.py"
"$PY" "$SCENE_DIR/source/u_wm.py" --events "$PREP/events" --wm-steps 30000 --h-steps 150000 --h-width 2048 --h-goals walk --h-walk-pool 300000 --h-absdiff --event-pos-only --rest-support --out "$OUT/wm" > "$OUT/wm.log" 2>&1 &
WM_PID=$!
trap 'kill "$WM_PID" 2>/dev/null || true' EXIT
"$PY" "$SCENE_DIR/source/u_skill.py" --cache "$PREP/cache/scene-play-v0" --events "$PREP/events" --train-frames 0 --val-frames 0 --steps 40000 --out "$OUT/skill" > "$OUT/skill.log" 2>&1
wait "$WM_PID"
trap - EXIT
LOOP=("$PY" "$SCENE_DIR/source/u_closed_loop.py" --env scene-v0 --model "$OUT/wm/u_model.pt" --state-layout "$PREP/front/layout.json" --skill "$OUT/skill/u_skill.pt" --events "$PREP/events" --cache "$PREP/cache/scene-play-v0" --workers 5 --max-expansions 20000)
"${LOOP[@]}" --episodes 1 --seed 0 --out "$OUT/loop_dev"
printf 'SCENE_TRAIN_AND_PILOT_COMPLETE\n'
