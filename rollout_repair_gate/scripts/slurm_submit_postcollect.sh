#!/usr/bin/env bash
#SBATCH --job-name=rrg_submit_post
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_submit_post_%j.out
set -euo pipefail

PROJECT=/home/nhatnc129/nhat.nc/jepa-research/rollout_repair_gate
cd "$PROJECT"
submit_retry() {
  local output
  # Slurm occasionally rejects submissions transiently while large arrays are
  # retiring. Keep the lightweight controller alive long enough to bridge that
  # condition; downstream afterok dependencies still prevent partial results
  # from being analysed.
  for attempt in $(seq 1 200); do
    if output=$(sbatch --parsable "$@"); then
      echo "$output"
      return 0
    fi
    sleep 15
  done
  return 1
}

SMOKE=$(submit_retry scripts/slurm_03_train_smoke.sh)
TRAIN=$(submit_retry --dependency="afterok:$SMOKE" scripts/slurm_03_train.sh)
FIXED=$(submit_retry --dependency="afterok:$TRAIN" scripts/slurm_04_fixed_eval.sh)
FRESH=$(submit_retry --dependency="afterok:$TRAIN" scripts/slurm_04_fresh_eval.sh)
ANALYZE=$(submit_retry --dependency="afterok:$FIXED:$FRESH" scripts/slurm_05_analyze.sh)
echo "TRAIN_SMOKE_JOB=$SMOKE"
echo "TRAIN_JOB=$TRAIN"
echo "FIXED_EVAL_JOB=$FIXED"
echo "FRESH_EVAL_JOB=$FRESH"
echo "ANALYSIS_JOB=$ANALYZE"
