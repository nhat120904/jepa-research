#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_parallel_aggregate
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_parallel_aggregate_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}
CLOSED_ID=${2:?closed array id}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_parallel_aggregate_$SLURM_JOB_ID"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
"$PY" "$CODE/scripts/cta_round3.py" --mode aggregate --run "$RUN" \
  --closed-run "$ROOT/cta_parallel_closed_$CLOSED_ID"
