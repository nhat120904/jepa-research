#!/usr/bin/env bash
#SBATCH --job-name=event_scene_plumb
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/event_scene_plumb_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
OUT="$PROJECT/outputs/scene_gate0/plumbing/${SLURM_JOB_ID}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/scene_core.py" \
  "$PROJECT/scripts/run_scene_gate0.py" \
  "$PROJECT/scripts/slurm_scene_gate0_plumbing.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/run_scene_gate0.py" \
  --task-id 4 --reset-seed 73004 --budgets 7 --horizon 2 --max-decisions 1 \
  --out-dir "$OUT/task4_seed73004"
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/run_scene_gate0.py" \
  --task-id 5 --reset-seed 73005 --budgets 7 --horizon 2 --max-decisions 1 \
  --out-dir "$OUT/task5_seed73005"

