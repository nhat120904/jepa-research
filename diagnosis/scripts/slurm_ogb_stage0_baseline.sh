#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_baseline_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch

cd "$DIAG/external/stable-worldmodel/scripts/plan"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "repo_commit=$(git -C "$REPO" rev-parse HEAD)"
echo "stable_worldmodel_commit=$(git -C "$DIAG/external/stable-worldmodel" rev-parse HEAD)"
sha256sum "$DIAG/scripts/slurm_ogb_stage0_baseline.sh" eval_wm.py config/cube.yaml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" eval_wm.py --config-name=cube \
  policy=quentinll/lewm-cube \
  seed=42 \
  eval.num_eval=50 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  solver.num_samples=300 \
  solver.n_steps=30 \
  solver.topk=30 \
  solver.var_scale=1.0 \
  solver.batch_size=1 \
  output.filename=ogb_stage0_baseline.txt

echo "===== OGB_STAGE0_BASELINE_DONE ===== $(date -u +%FT%TZ)"
