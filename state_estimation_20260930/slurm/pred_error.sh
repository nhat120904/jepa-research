#!/usr/bin/env bash
#SBATCH --job-name=se_prederr
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/state_estimation/logs/%x_%A_%a.out
# Offline rollout-error decomposition by start state (array index -> task).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/state_estimation_20260930/slurm/env.sh
TASKS=(reacher pusht cube tworoom)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
N=${N:-600}
OUT=$RUN_ROOT/pred_error_${SLURM_ARRAY_JOB_ID}
record_source "$OUT"
"$PY" "$PROJECT/scripts/pred_error.py" --task "$TASK" --n "$N" --out "$OUT/$TASK.json"
