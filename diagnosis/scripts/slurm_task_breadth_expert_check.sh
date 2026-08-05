#!/usr/bin/env bash
#SBATCH --job-name=jepa_taskcheck
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_breadth_expert_check_%j.out
# Gate 1 for the task-breadth extension (docs/plans/<date>-generality-extension
# -design.md). No model, no CEM search -- just scripted-expert rollouts, so this
# is cheap, but MuJoCo rendering still needs the EGL context, hence a GPU node.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
EPISODES=${TASKCHECK_EPISODES:-16}
SEED0=${TASKCHECK_SEED0:-70000}
TASKS=(mw-door-open mw-drawer-close mw-button-press mw-window-close mw-assembly mw-peg-insert-side)

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} seed0=$SEED0 episodes=$EPISODES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "commit=$(git rev-parse HEAD)"

"$PY" scripts/70_task_breadth_expert_check.py \
  --tasks "${TASKS[@]}" --episodes "$EPISODES" --seed0 "$SEED0" \
  --out results/task_breadth_expert_check.csv \
  --out-md results/task_breadth_expert_check.md

echo "TASK_BREADTH_EXPERT_CHECK_DONE $(date)"
