#!/usr/bin/env bash
#SBATCH --job-name=perd_submit_eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_submit_eval_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
cd "$REPO"

submit_with_retry() {
  local output
  for attempt in $(seq 1 20); do
    if output=$(sbatch --parsable "$@" 2>&1); then
      echo "$output"
      return 0
    fi
    echo "attempt=$attempt submission_failed=$output" >&2
    sleep 30
  done
  return 1
}

EVAL=$(submit_with_retry --dependency=afterok:${SLURM_JOB_ID} "$PROJECT/scripts/slurm_03_eval.sh")
ANALYZE=$(submit_with_retry --dependency="afterok:$EVAL" "$PROJECT/scripts/slurm_04_analyze.sh")
echo "eval=$EVAL"
echo "analyze=$ANALYZE"
