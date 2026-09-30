#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_r2_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_r2_%A_%a.out
# Array 0 = matched continuation, lambda=0; 1 = method, lambda=0.1.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release directory}
PARENT=${2:?round 1 training directory}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
LAMBDAS=(0 0.1)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
export CTA_TAG=r2_codesign
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
"$PY" "$CODE/scripts/cta_round2.py" --parent "$PARENT" --run "$RUN" --lam "$LAMBDA"
