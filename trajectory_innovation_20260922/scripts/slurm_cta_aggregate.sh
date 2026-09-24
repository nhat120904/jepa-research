#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_agg_%j.out
# Usage: CTA_TAG=r0 sbatch slurm_cta_aggregate.sh <first_root> <count> <cta_cl_dir> [more cta_cl dirs]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
FIRST=${1:?first}; COUNT=${2:?count}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_agg_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
"$PY" "$CODE/scripts/cta_aggregate.py" --run "${@:3}" --out "$RUN" --first "$FIRST" --count "$COUNT"
