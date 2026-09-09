#!/usr/bin/env bash
#SBATCH --job-name=handoff_a1
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/handoff_a1_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch $0 /absolute/path/to/best.pt" >&2
  exit 2
fi
CHECKPOINT=$1
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/skill_handoff_wm"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
DATA=/mnt/data/vhoangth2/datasets/ogbench_data
OUT="$PROJECT/outputs/a1_smoke/${SLURM_JOB_ID}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/maze.py" \
  "$PROJECT/policy.py" \
  "$PROJECT/sim.py" \
  "$PROJECT/switching.py" \
  "$PROJECT/scripts/eval_a1.py" \
  "$CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_a1.py" \
  --dataset-name antmaze-medium-navigate-v0 \
  --dataset-dir "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT" \
  --task-ids 1,2 --seeds 92001,92002 \
  --durations 10,20,40,80 --probe-steps 40 --primitive-budget 1000
