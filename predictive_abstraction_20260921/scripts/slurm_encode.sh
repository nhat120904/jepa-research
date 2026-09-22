#!/usr/bin/env bash
#SBATCH --job-name=pa_dinov3_encode
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/predictive_abstraction/logs/encode_%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Use sbatch" >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/predictive_abstraction_20260921
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/predictive_abstraction/encode_${SLURM_JOB_ID}
[[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 2; }
mkdir -p "$RUN/code"
cp -r "$PROJECT/pa_wm" "$PROJECT/configs" "$PROJECT/docs" "$RUN/code/"
cp "$PROJECT/scripts/slurm_encode.sh" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/SHA256SUMS"
sha256sum /mnt/data/nhatnc129/jepa/ossckpt/dinov3/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth > "$RUN/CHECKPOINT_SHA256"
export PYTHONPATH="$RUN/code"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PY" -m pa_wm.encode --config "$RUN/code/configs/encode_dinov3.json" \
  --run-dir "$RUN/output"
