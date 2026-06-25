#!/usr/bin/env bash
# Track 2: isolated RoboCasa/RoboSuite install (does NOT touch the working .venv).
# Builds .venv-robocasa with the jepa-wms deps + the two Basile-Terv forks + the
# ~20GB kitchen assets, so a closed-loop RoboCasa Pick harness can run later.
# Best-effort: clone/install stages are fatal, the 20GB asset download is not (a
# partial env is still useful to diagnose). Verbose logging for the background run.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
LOG() { echo "[robocasa-setup $(date +%H:%M:%S)] $*"; }
set -x

VENV=.venv-robocasa
EXT=external

# 1. isolated venv (python 3.10 per upstream pin) + jepa-wms deps
[ -d "$VENV" ] || uv venv "$VENV" --python 3.10
set +x; LOG "venv ready: $VENV"; set -x
VPY="$VENV/bin/python"
uv pip install --python "$VPY" -e "$EXT/jepa-wms" 2>&1 | tail -20

# 2. robosuite fork (robocasa-dev)
if [ ! -d "$EXT/robosuite" ]; then
  git clone --depth 1 https://github.com/Basile-Terv/robosuite.git "$EXT/robosuite"
fi
uv pip install --python "$VPY" -e "$EXT/robosuite" 2>&1 | tail -20

# 3. robocasa fork
if [ ! -d "$EXT/robocasa" ]; then
  git clone --depth 1 https://github.com/Basile-Terv/robocasa.git "$EXT/robocasa"
fi
uv pip install --python "$VPY" -e "$EXT/robocasa" 2>&1 | tail -20

set +x
LOG "import check:"
"$VPY" -c "import robosuite, robocasa; print('robosuite', robosuite.__version__); print('robocasa OK')" \
  && LOG "IMPORTS OK" || LOG "IMPORT FAILED (see above)"

# 4. assets (~20GB) — non-fatal; can be re-run
LOG "downloading kitchen assets (~20GB, non-fatal)..."
( cd "$EXT/robocasa" && "../../$VENV/bin/python" robocasa/scripts/download_kitchen_assets.py ) \
  && LOG "ASSETS OK" || LOG "ASSET DOWNLOAD incomplete (re-runnable)"
( cd "$EXT/robocasa" && "../../$VENV/bin/python" robocasa/scripts/setup_macros.py ) \
  && LOG "MACROS OK" || LOG "MACROS step failed"

LOG "===== ROBOCASA_SETUP_DONE ====="
