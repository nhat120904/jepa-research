#!/usr/bin/env bash
#SBATCH --job-name=handoff_a1full
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/handoff_a1full_%j.out
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
OUT="$PROJECT/outputs/a1_screen/${SLURM_JOB_ID}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/maze.py" \
  "$PROJECT/policy.py" \
  "$PROJECT/sim.py" \
  "$PROJECT/switching.py" \
  "$PROJECT/scripts/eval_a1.py" \
  "$PROJECT/scripts/analyze_a1.py" \
  "$PROJECT/scripts/slurm_a1_screen.sh" \
  "$CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_a1.py" \
  --dataset-name antmaze-medium-navigate-v0 \
  --dataset-dir "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT/eval" \
  --task-ids 1,2,3,4,5 \
  --seeds 93001,93002,93003,93004,93005,93006,93007,93008,93009,93010,93011,93012,93013,93014,93015,93016,93017,93018,93019,93020 \
  --durations 10,20,40,80 --probe-steps 40 --primitive-budget 1000

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_a1.py" \
  --episodes "$OUT/eval/episodes.jsonl" \
  --out "$OUT/gate.json" \
  --gate-points 10
