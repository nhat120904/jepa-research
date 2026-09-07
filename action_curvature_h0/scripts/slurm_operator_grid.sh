#!/usr/bin/env bash
#SBATCH --job-name=acm_oper
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_oper_%A_%a.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
SWM="$REPO/diagnosis/external/stable-worldmodel"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

SEEDS=(20260714 7 11 13 17 19 23 29 31 37)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?}]}
OUT="$PROJECT/outputs/operator/seed_${SEED}"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} PLAN_SEED=$SEED $(date -u +%FT%TZ)"

# Same checkpoint, same plan seed, same episodes, same 300/30/30 budget.
# The ONLY difference is which scored candidate gets executed:
#   elite_mean = deployed CEM (refit mean)      -- already measured, rerun here
#                                                  so both arms share this job
#   best_seen  = the lowest-cost candidate ever scored (iCEM semantics)
cd "$SWM/scripts/plan"
echo "=== elite_mean (deployed) $(date -u +%FT%TZ) ==="
"$PY" eval_wm.py --config-name=cube policy=acm_original seed="$SEED" \
  > "$OUT/elite_mean.log" 2>&1 || { echo FAILED; tail -20 "$OUT/elite_mean.log"; exit 1; }
grep -oE "'success_rate': [0-9.]+" "$OUT/elite_mean.log" | tail -1

echo "=== best_seen $(date -u +%FT%TZ) ==="
"$PY" eval_wm.py --config-name=cube policy=acm_original seed="$SEED" \
  solver._target_=action_curvature_h0.solvers.BestEliteCEMSolver \
  > "$OUT/best_seen.log" 2>&1 || { echo FAILED; tail -25 "$OUT/best_seen.log"; exit 1; }
grep -oE "'success_rate': [0-9.]+" "$OUT/best_seen.log" | tail -1

"$PY" "$PROJECT/scripts/collect_operator.py" --run-dir "$OUT" --seed "$SEED"
