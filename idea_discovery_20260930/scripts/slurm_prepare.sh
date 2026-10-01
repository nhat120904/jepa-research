#!/usr/bin/env bash
#SBATCH --job-name=idea_assets
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/idea_discovery_20260930/logs/assets_%j.out
set -euo pipefail
ROOT=/home/nhatnc129/nhat.nc/jepa-research/idea_discovery_20260930
RUNROOT=/mnt/data/nhatnc129/jepa/idea_discovery_20260930
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
mkdir -p "$RUNROOT/extra"
if ! "$PY" -c 'import gdown' >/dev/null 2>&1; then
    /home/nhatnc129/.local/bin/uv pip install --target "$RUNROOT/extra" gdown
fi
export PYTHONPATH="$RUNROOT/extra${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$ROOT/scripts/prepare_assets.py" --root "$RUNROOT/assets"
