#!/usr/bin/env bash
# Environment bootstrap — run once per machine.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Clone the canonical jepa-wms repo (checkpoints + datasets + baselines).
if [[ ! -d external/jepa-wms ]]; then
    git clone https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
fi

# 2. Sync the upstream environment.
pushd external/jepa-wms
# facebookresearch/jepa-wms currently points uv at Metaworld's removed
# `master` branch. The V3 pyproject branch preserves the API this repo uses
# while avoiding the current `mujoco==3.3.0` resolver conflict on Metaworld main.
if grep -q 'Metaworld.git", branch = "\(master\|main\)"' pyproject.toml; then
    sed -i 's#Metaworld.git", branch = "\(master\|main\)"#Metaworld.git", branch = "v3_pyproject"#' pyproject.toml
fi

if [[ -f uv.lock ]]; then
    uv sync
else
    if [[ ! -d .venv ]]; then
        uv venv
    fi
    uv pip install -e .
fi
popd

# 3. Install diagnostic deps on top of the upstream uv environment. `uv venv`
# intentionally does not install pip, so use `uv pip` instead of system `pip`.
uv pip install --python external/jepa-wms/.venv/bin/python -e .

# Make the standard local activation path work from this directory too.
if [[ ! -e .venv ]]; then
    ln -s external/jepa-wms/.venv .venv
fi

echo "[OK] Environment ready. Run scripts/02_download_checkpoints.py next."
