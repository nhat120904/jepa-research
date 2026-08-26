#!/usr/bin/env bash
#SBATCH --job-name=acm_cemcol
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=64-127%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_cemcol_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PERD/scripts/collect_populations.py" "$PROJECT/scripts/slurm_cem_collect.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Deployed cube CEM budget, verified against scripts/plan/config/solver/cem.yaml
# (num_samples 300, n_steps 30, topk 30) and config/cube.yaml (horizon 5,
# action_block 5).  Held-out orders 64-127; the shared population every arm is
# scored on.  The index-0 pre-refit gate runs per shard, not once.
echo "--- collect start $(date -u +%FT%TZ) ---"
"$STAGE0_ROOT/.venv/bin/python" "$PERD/scripts/collect_populations.py" \
  --snapshot-index "${SLURM_ARRAY_TASK_ID:?}" --manifest "$PERD/outputs/h0/manifest.json" \
  --num-samples 300 --cem-steps 30 --topk 30 \
  --record-step 0 --record-step 29 \
  --out-dir "$PROJECT/outputs/cem_populations"
echo "--- collect end $(date -u +%FT%TZ) ---"

"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/check_population_index0.py" \
  --populations "$PROJECT/outputs/cem_populations/snapshot_$(printf %03d ${SLURM_ARRAY_TASK_ID})/populations.npz" \
  --num-samples 300 --topk 30
