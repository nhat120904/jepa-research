#!/usr/bin/env bash
#SBATCH --job-name=scene_h1_csmoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h1_csmoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
OUT="$PROJECT/outputs/scene_h1/data_smoke/${SLURM_JOB_ID}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_core.py" "$PROJECT/scene_learning.py" \
  "$PROJECT/scripts/collect_scene_h1.py" "$PROJECT/scripts/slurm_scene_h1_collect_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/collect_scene_h1.py" \
  --task-id 4 --reset-seed 80400 --split smoke --out-dir "$OUT/task4"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/collect_scene_h1.py" \
  --task-id 5 --reset-seed 80500 --split smoke --out-dir "$OUT/task5"

