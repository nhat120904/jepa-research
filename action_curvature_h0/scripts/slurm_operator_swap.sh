#!/usr/bin/env bash
#SBATCH --job-name=acm_opsw
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_opsw_%A_%a.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
SWM="$REPO/diagnosis/external/stable-worldmodel"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

SEEDS=(20260714 7 11 13 17 19 23 29 31 37)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?}]}
OUT="$PROJECT/outputs/operator_swap/seed_${SEED}"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} PLAN_SEED=$SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/solvers/best_candidate_cem.py"

# Same checkpoint (the released one), same plan seed, same episodes, same CEM
# search.  The ONLY difference between the two arms is which action sequence is
# executed: the refit elite mean, or the best candidate the search ever scored.
cd "$SWM/scripts/plan"
echo "=== mean (deployed) $(date -u +%FT%TZ) ==="
"$PY" eval_wm.py --config-name=cube policy=acm_original seed="$SEED" \
  > "$OUT/exec_mean.log" 2>&1 || { echo FAILED; tail -20 "$OUT/exec_mean.log"; exit 1; }
grep -oE "'success_rate': [0-9.]+" "$OUT/exec_mean.log" | tail -1

echo "=== best candidate $(date -u +%FT%TZ) ==="
"$PY" eval_wm.py --config-name=cube policy=acm_original seed="$SEED" \
  solver._target_=action_curvature_h0.solvers.BestCandidateCEMSolver \
  > "$OUT/exec_best.log" 2>&1 || { echo FAILED; tail -20 "$OUT/exec_best.log"; exit 1; }
grep -oE "'success_rate': [0-9.]+" "$OUT/exec_best.log" | tail -1

"$PY" "$PROJECT/scripts/collect_operator_swap.py" --run-dir "$OUT" --seed "$SEED"
