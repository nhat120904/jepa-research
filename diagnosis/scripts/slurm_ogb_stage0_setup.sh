#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_setup_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
VENV="$STAGE0_ROOT/.venv"
ARCHIVE="$STAGE0_ROOT/downloads/cube_single_expert.tar.zst"
DATASET="$STAGE0_ROOT/datasets/ogbench/cube_single_expert.h5"

mkdir -p "$STAGE0_ROOT/downloads" "$STAGE0_ROOT/datasets/ogbench"
mkdir -p /mnt/data/nhatnc129/jepa_runs/logs

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "repo_commit=$(git -C "$REPO" rev-parse HEAD)"
echo "lewm_commit=$(git -C "$DIAG/external/le-wm" rev-parse HEAD)"
echo "stable_worldmodel_commit=$(git -C "$DIAG/external/stable-worldmodel" rev-parse HEAD)"
echo "stable_pretraining_commit=$(git -C "$DIAG/external/stable-pretraining" rev-parse HEAD)"

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python=3.10 "$VENV"
fi

uv pip install --python "$VENV/bin/python" \
  -e "$DIAG/external/stable-pretraining" \
  -e "$DIAG/external/stable-worldmodel" \
  ogbench h5py hdf5plugin opencv-python-headless imageio imageio-ffmpeg \
  pygame pymunk shapely

# The cluster driver supports CUDA 12.8.  PyPI's unconstrained torch wheel is
# currently CUDA 13 and cannot initialize here, so use the official cu126 pair.
uv pip install --python "$VENV/bin/python" \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$VENV/bin/python" transformers==4.57.1

if [[ ! -f "$DATASET" ]]; then
  curl --fail --location --http1.1 --retry 20 --retry-all-errors \
    --retry-delay 10 --retry-max-time 7200 --continue-at - \
    --output "$ARCHIVE" \
    https://huggingface.co/datasets/quentinll/lewm-cube/resolve/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst
  sha256sum "$ARCHIVE"
  EXTRACT_DIR="$STAGE0_ROOT/extracted_cube"
  mkdir -p "$EXTRACT_DIR"
  tar --zstd -xf "$ARCHIVE" -C "$EXTRACT_DIR"
  SOURCE=$(find "$EXTRACT_DIR" -type f -name 'cube_single_expert.h5' -print -quit)
  if [[ -z "$SOURCE" ]]; then
    echo "cube_single_expert.h5 not found after extraction" >&2
    exit 2
  fi
  mv "$SOURCE" "$DATASET"
fi

export STABLEWM_HOME="$STAGE0_ROOT"
"$VENV/bin/python" - <<'PY'
import stable_pretraining
import stable_worldmodel
print("stable_pretraining", stable_pretraining.__file__)
print("stable_worldmodel", stable_worldmodel.__file__)
PY

ls -lh "$DATASET"
echo "===== OGB_STAGE0_SETUP_DONE ===== $(date -u +%FT%TZ)"
