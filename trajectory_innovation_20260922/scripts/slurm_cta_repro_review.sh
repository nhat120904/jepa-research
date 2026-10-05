#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_repro_review
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_repro_review_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 1
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN=/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_repro_review_${SLURM_JOB_ID}
CODE="$RUN/code"
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$CODE/"
mkdir "$CODE/scripts"
cp "$PROJECT/scripts/cta_repro_review.py" "$CODE/scripts/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SOURCE_SHA256SUMS"
export WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
"$PY" "$CODE/scripts/cta_repro_review.py" --out "$RUN/results"
