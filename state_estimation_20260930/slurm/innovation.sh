#!/usr/bin/env bash
#SBATCH --job-name=se_innov
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/state_estimation/logs/%x_%A_%a.out
# Innovation predictability of history-3 rollout errors (array index -> task).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/state_estimation_20260930/slurm/env.sh
TASKS=(reacher pusht cube tworoom)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
N=${N:-3000}
OUT=$RUN_ROOT/innovation_${SLURM_ARRAY_JOB_ID}
record_source "$OUT"
"$PY" "$PROJECT/scripts/innovation.py" --task "$TASK" --n "$N" --out "$OUT/$TASK.json"
