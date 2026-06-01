#!/usr/bin/env bash
# Environment bootstrap — run once per machine.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Clone the canonical jepa-wms repo (checkpoints + datasets + baselines).
if [[ ! -d external/jepa-wms ]]; then
    git clone https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
fi

# 2. Sync the upstream environment (it ships a uv.lock).
pushd external/jepa-wms
if [[ -f uv.lock ]]; then
    uv sync
else
    uv venv && uv pip install -e .
fi

# Activate the same venv for the diagnostic pipeline.
source .venv/bin/activate
popd

# 3. Install diagnostic deps on top.
pip install -e .

echo "[OK] Environment ready. Run scripts/02_download_checkpoints.py next."
