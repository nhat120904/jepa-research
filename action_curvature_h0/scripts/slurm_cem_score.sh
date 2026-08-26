#!/usr/bin/env bash
#SBATCH --job-name=acm_cemsc
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=64-127%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_cemsc_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:?} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/score_cem_arms.py" "$PROJECT/scripts/slurm_cem_score.sh"

# All four arms in one task: same simulator, same population, candidate hashes
# checked equal before anything is written.  The scorer re-scores and refits
# only; it never samples.  Every snapshot 64-127 is scored; the locked viability
# filter (sixteenth amendment) is applied at analysis, so the excluded snapshots
# still yield their corroboration data.
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/score_cem_arms.py" \
  --snapshot-index "${SLURM_ARRAY_TASK_ID}" \
  --manifest "$PERD/outputs/h0/manifest.json" \
  --populations-dir "$PROJECT/outputs/cem_populations" \
  --arm original \
  --arm "lam0_seed0=$PROJECT/outputs/dev/lam0_seed0/checkpoint.pt" \
  --arm "lam0_seed1=$PROJECT/outputs/dev/lam0_seed1/checkpoint.pt" \
  --arm "lam0_seed2=$PROJECT/outputs/dev/lam0_seed2/checkpoint.pt" \
  --topk 30 --out-dir "$PROJECT/outputs/cem_score"
