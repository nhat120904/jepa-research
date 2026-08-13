#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_pinmodel
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_pin_model_%j.out
set -euo pipefail

STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
VENV="$STAGE0_ROOT/.venv"
MODEL_REV=b0747c5002e86d2ce8f3cd8178004b97524c587d
MODEL_DIR="$STAGE0_ROOT/checkpoints/models--quentinll--lewm-cube"
mkdir -p "$MODEL_DIR" /mnt/data/nhatnc129/jepa_runs/logs

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
for filename in config.json weights.pt; do
  curl --fail --location --retry 8 --retry-delay 10 \
    --output "$MODEL_DIR/$filename.tmp" \
    "https://huggingface.co/quentinll/lewm-cube/resolve/$MODEL_REV/$filename"
  mv "$MODEL_DIR/$filename.tmp" "$MODEL_DIR/$filename"
done

sha256sum "$MODEL_DIR/config.json" "$MODEL_DIR/weights.pt"
ls -lh "$MODEL_DIR/config.json" "$MODEL_DIR/weights.pt"
uv pip freeze --python "$VENV/bin/python" | tee "$STAGE0_ROOT/environment-freeze.txt"
echo "model_revision=$MODEL_REV"
echo "===== OGB_STAGE0_PIN_MODEL_DONE ===== $(date -u +%FT%TZ)"
