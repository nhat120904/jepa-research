#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_train_%j.out
# Usage: CTA_TAG=r0 sbatch slurm_cta_train.sh <cta_feat_dir> [cta_train.py args, e.g. --seed 1 --lam 0.1 --steps3 6000]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
FEAT=${1:?features dir}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_train_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
"$PY" "$CODE/scripts/cta_train.py" --run "$RUN" --features "$FEAT" "${@:2}"
