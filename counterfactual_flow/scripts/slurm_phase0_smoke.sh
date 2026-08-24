#!/usr/bin/env bash
#SBATCH --job-name=cf_flow_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_flow_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/counterfactual_flow"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT_INDEX=${CFLOW_SMOKE_SNAPSHOT:-0}
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} SNAPSHOT=$SNAPSHOT_INDEX $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/scripts/mine_ogb_counterfactuals.py" \
  "$PROJECT/scripts/slurm_phase0_smoke.sh" \
  "$REPO/diagnosis/scripts/76_ogb_true_endpoint_corrected.py"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/mine_ogb_counterfactuals.py" \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --out-dir "$PROJECT/outputs/ogbench_cube_phase0/smoke_$SNAPSHOT_INDEX"

echo "===== CFLOW_PHASE0_SMOKE_DONE ===== $(date -u +%FT%TZ)"
