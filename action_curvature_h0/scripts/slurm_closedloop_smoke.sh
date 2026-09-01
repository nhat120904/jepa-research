#!/usr/bin/env bash
#SBATCH --job-name=acm_clsmoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_clsmoke_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
SWM="$REPO/diagnosis/external/stable-worldmodel"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"

CKPT_ROOT="$STAGE0_ROOT/checkpoints"
"$PY" "$PROJECT/scripts/prepare_closedloop_ckpts.py" \
  --released "$CKPT_ROOT/models--quentinll--lewm-cube" \
  --arm acm_original \
  --arm "acm_lam0_seed0=$PROJECT/outputs/dev/lam0_seed0/checkpoint.pt" \
  --arm "acm_lam0_seed1=$PROJECT/outputs/dev/lam0_seed1/checkpoint.pt" \
  --arm "acm_lam0_seed2=$PROJECT/outputs/dev/lam0_seed2/checkpoint.pt" \
  --out-root "$CKPT_ROOT"

# One arm, one plan seed, through the OFFICIAL evaluator with the OFFICIAL cube
# config -- nothing forked.  Times the closed-loop protocol before any grid.
cd "$SWM/scripts/plan"
echo "--- eval start $(date -u +%FT%TZ) ---"
"$PY" eval_wm.py --config-name=cube policy=acm_original seed=20260714 \
  2>&1 | tail -40
echo "--- eval end $(date -u +%FT%TZ) ---"
