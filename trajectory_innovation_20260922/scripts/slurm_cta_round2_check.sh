#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_r2_check
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_r2_check_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release directory}
PARENT=${2:?round 1 training directory}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_r2_check_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG=r2_check WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
for LAMBDA in 0 0.1; do
  "$PY" "$CODE/scripts/cta_round2.py" --parent "$PARENT" --run "$RUN/lam${LAMBDA}" \
    --lam "$LAMBDA" --source-steps 2 --src-block 1 --wm-block 1 --align-steps 2 \
    --warmup 1 --eval-every 2 --eval-n 2 --n-train 8 --n-dev 2 --samples 0 --device cpu --smoke
done
echo ROUND2_CHECK_OK
