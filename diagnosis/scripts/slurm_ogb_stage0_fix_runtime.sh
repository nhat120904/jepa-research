#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_runtime
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_fix_runtime_%j.out
set -euo pipefail

STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
VENV="$STAGE0_ROOT/.venv"
mkdir -p /mnt/data/nhatnc129/jepa_runs/logs

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
uv pip install --python "$VENV/bin/python" \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$VENV/bin/python" \
  transformers==4.57.1 pygame pymunk shapely

"$VENV/bin/python" - <<'PY'
import pygame
import pymunk
import shapely
import torch
import torchvision
import transformers

print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("pygame", pygame.version.ver)
print("pymunk", pymunk.version)
print("shapely", shapely.__version__)
assert torch.version.cuda == "12.6"
PY

export STABLEWM_HOME="$STAGE0_ROOT"
"$VENV/bin/python" - <<'PY'
import stable_worldmodel as swm

model = swm.wm.utils.load_pretrained("quentinll/lewm-cube")
print("checkpoint_load_ok", type(model).__module__, type(model).__name__)
print("model_parameters", sum(p.numel() for p in model.parameters()))
PY

uv pip freeze --python "$VENV/bin/python" | tee "$STAGE0_ROOT/environment-freeze.txt"
echo "===== OGB_STAGE0_FIX_RUNTIME_DONE ===== $(date -u +%FT%TZ)"
