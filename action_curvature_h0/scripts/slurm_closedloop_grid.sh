#!/usr/bin/env bash
#SBATCH --job-name=acm_cloop
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_cloop_%A_%a.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
SWM="$REPO/diagnosis/external/stable-worldmodel"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled

SEEDS=(20260714 7 11 13 17 19 23 29 31 37)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?}]}
OUT="$PROJECT/outputs/closedloop/seed_${SEED}"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} PLAN_SEED=$SEED $(date -u +%FT%TZ)"

# All four arms at the SAME plan seed, so they see the same episodes and the
# comparison is paired.  The episode list each run selects is captured and
# checked equal across arms afterwards; a mismatch invalidates the pairing.
cd "$SWM/scripts/plan"
for ARM in acm_original acm_lam0_seed0 acm_lam0_seed1 acm_lam0_seed2; do
  echo "=== $ARM seed=$SEED $(date -u +%FT%TZ) ==="
  "$PY" eval_wm.py --config-name=cube policy="$ARM" seed="$SEED" \
    > "$OUT/${ARM}.log" 2>&1 || { echo "$ARM FAILED"; tail -20 "$OUT/${ARM}.log"; exit 1; }
  grep -oE "'success_rate': [0-9.]+" "$OUT/${ARM}.log" | tail -1
done

"$PY" "$PROJECT/scripts/collect_closedloop.py" --run-dir "$OUT" --seed "$SEED"
