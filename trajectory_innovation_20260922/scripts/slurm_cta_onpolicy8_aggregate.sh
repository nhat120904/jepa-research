#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_onpol8_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_onpol8_agg_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=${1:?immutable source release}
CLOSED_ID=${2:?closed array id}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
RUN="$TI/cta_onpolicy8_aggregate_${SLURM_JOB_ID}"
CODE="$RUN/code"
HASHES="$RUN/CODE_SHA256SUMS"
export WANDB_MODE=disabled CTA_TAG=cta_onpolicy8_aggregate
source "$PROJECT/scripts/cta_env.sh"
snapshot "$HASHES"
"$PY" "$CODE/scripts/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" \
  --closed-run "$TI/cta_onpolicy8_closed_$CLOSED_ID" \
  --pairs CTA8O-DIRECT8O,CTA8O-ENDPOINT8O,CTA8O-CTA4,CTA8O-GEOM8 --expect-roots 2200 2399
