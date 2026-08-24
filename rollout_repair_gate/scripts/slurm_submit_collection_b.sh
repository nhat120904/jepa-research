#!/usr/bin/env bash
#SBATCH --job-name=rrg_submit_b
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_submit_b_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
cd "$PROJECT"
for attempt in $(seq 1 20); do
  if JOB_B=$(sbatch --parsable scripts/slurm_01_collect_b.sh); then
    echo "COLLECTION_B_JOB=$JOB_B"
    POST=$(sbatch --parsable --dependency="afterok:$JOB_B" scripts/slurm_submit_postcollect.sh)
    echo "POSTCOLLECT_SUBMITTER_JOB=$POST"
    exit 0
  fi
  echo "submission attempt $attempt failed; retrying in 15 seconds" >&2
  sleep 15
done
exit 1

