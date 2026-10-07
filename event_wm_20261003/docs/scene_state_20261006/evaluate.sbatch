#!/usr/bin/env bash
#SBATCH --job-name=ew_sceval
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$SCENE_DIR/source"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP="$SCENE_DIR/prep_57771"
MODEL_DIR="${MODEL_DIR:-$SCENE_DIR/job_57772}"
OUT="$SCENE_DIR/eval_$SLURM_JOB_ID"
mkdir "$OUT"
cd "$SCENE_DIR"
find source -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_SHA256SUMS"
"$PY" "$SCENE_DIR/source/u_closed_loop.py" --env scene-v0 --model "$MODEL_DIR/wm/u_model.pt" --state-layout "$PREP/front/layout.json" --skill "$MODEL_DIR/skill/u_skill.pt" --events "$PREP/events" --cache "$PREP/cache/scene-play-v0" --workers 5 --max-expansions 20000 --episodes 20 --seed 3 --out "$OUT/loop_seed3"
