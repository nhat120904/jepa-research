#!/usr/bin/env bash
#SBATCH --job-name=erwm_e0_setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/erwm_e0_setup_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME=/mnt/data/nhatnc129/jepa/erwm_e0
VENV="$RUNTIME/.venv"

mkdir -p "$RUNTIME" /mnt/data/nhatnc129/jepa_runs/logs
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python=3.10 "$VENV"
fi
uv pip install --python "$VENV/bin/python" \
  mani_skill==3.0.1 torch==2.7.0 numpy==1.26.4

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
"$VENV/bin/python" - <<'PY'
import gymnasium
import mani_skill
import sapien
import torch

print("mani_skill", getattr(mani_skill, "__version__", "unknown"))
print("sapien", getattr(sapien, "__version__", "unknown"))
print("torch", torch.__version__)
print("gymnasium", gymnasium.__version__)
PY

