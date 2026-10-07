#!/usr/bin/env bash
#SBATCH --job-name=ew_puzloop
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPAIR_DIR/source_fast:$REPAIR_DIR/source_support"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
BASE=/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614
OUT="$REPAIR_DIR/job_$SLURM_JOB_ID"
mkdir -p "$OUT"
"$PY" -c 'import s_entities; print("STATE_IMPORT_OK", flush=True)'
"$PY" "$REPAIR_DIR/source_fast/u_closed_loop.py" --env puzzle-4x5-v0 --model "$REPAIR_DIR/job_57684/u_model.pt" --state-layout "$BASE/front/layout.json" --skill "$BASE/skill/u_skill.pt" --events "$BASE/events" --cache "$BASE/cache/puzzle-4x5-play-v0" --workers 5 --episodes 1 --seed 0 --max-expansions 20000 --out "$OUT/loop_seed0"
