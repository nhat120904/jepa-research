#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_geometry_e2e
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_geometry_e2e_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_geometry_e2e_$SLURM_JOB_ID"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export CTA_TAG=geometry_e2e_dev
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
RESUME_ARGS=()
if [[ -n "${2:-}" ]]; then
    RESUME_ARGS=(--resume-from "$2")
fi
"$PY" "$CODE/scripts/cta_geometry_e2e.py" --run "$RUN" \
    --parent "$ROOT/cta_train_54717" --features "$ROOT/cta_feat_54490" \
    --collection "$ROOT/cta_collect_54489" --prep "$PREP" --smoke "$SMOKE" "${RESUME_ARGS[@]}"
