#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_r2_compare
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_r2_compare_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release directory}
ARRAY=${2:?training array id}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_compare_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG=r2_compare WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
"$PY" "$CODE/scripts/cta_round2_compare.py" \
  --control "$ROOT/cta_r2_${ARRAY}_0" --method "$ROOT/cta_r2_${ARRAY}_1" --out "$RUN/comparison.json"
