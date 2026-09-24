#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_encode
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_encode_%j.out
# Usage: sbatch slurm_cta_encode.sh <cta_collect_run_dir>
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
COLLECT=${1:?collect run dir}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_feat_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
"$PY" "$CODE/scripts/cta_encode.py" --collect "$COLLECT" --out "$RUN" --smoke "$SMOKE"
