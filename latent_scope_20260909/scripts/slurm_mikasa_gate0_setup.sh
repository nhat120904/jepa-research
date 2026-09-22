#!/bin/bash
#SBATCH --job-name=mikasa_gate0_setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_setup_%j.out
#SBATCH --error=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_setup_%j.out

# Gate 0 setup: build an isolated ManiSkill3 environment. CPU only, no GPU held.
# Does not touch any existing checkout or venv used by earlier programmes.

set -euo pipefail

ROOT=/mnt/data/nhatnc129/jepa/mikasa_gate0
VENV="$ROOT/.venv"
# Cluster driver is 570.x => CUDA 12.8. Unpinned resolution pulls torch cu130,
# which fails at torch._C._cuda_init() before SAPIEN is ever touched (job 53280).
SPEC="mani-skill+torch-cu128"
MARKER="$VENV/.gate0_ready"

mkdir -p "$ROOT/logs" "$ROOT/outputs"

echo "[setup] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date -Is)"

if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$SPEC" ]; then
  echo "[setup] environment already built for spec '$SPEC'; skipping install"
else
  rm -f "$MARKER"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup] creating venv"
    uv venv --python 3.11 "$VENV"
  fi
  echo "[setup] installing mani-skill"
  uv pip install --python "$VENV/bin/python" "mani-skill"

  # Job 53281: an unpinned reinstall with --extra-index-url simply re-picked
  # torch 2.14.0+cu130 from PyPI. Try explicit candidates and verify each one,
  # rather than assuming any single spec yields a CUDA 12.x build.
  echo "[setup] replacing torch with a CUDA 12.x build"
  TORCH_SOURCE=""
  try_torch() {
    echo "[setup] trying torch spec: $*"
    if uv pip install --python "$VENV/bin/python" --reinstall-package torch "$@"; then
      if "$VENV/bin/python" -c "
import sys, torch
cuda = torch.version.cuda or ''
print('  -> torch', torch.__version__, 'cuda', cuda)
sys.exit(0 if cuda.split('.')[0] == '12' else 1)
"; then
        return 0
      fi
      echo "[setup]   rejected: not a CUDA 12.x build"
    else
      echo "[setup]   install failed"
    fi
    return 1
  }

  if try_torch "torch==2.8.0"; then
    TORCH_SOURCE="pypi-2.8.0"
  elif try_torch "torch==2.7.1"; then
    TORCH_SOURCE="pypi-2.7.1"
  elif try_torch --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128"; then
    TORCH_SOURCE="pytorch-index-cu128-2.8.0"
  else
    echo "[setup] FATAL: no candidate produced a CUDA 12.x torch build"
    exit 1
  fi
  echo "[setup] torch source: $TORCH_SOURCE"
fi

echo "[setup] verifying torch CUDA build and imports (no GPU needed)"
"$VENV/bin/python" - <<'PY'
import sys
import torch

cuda = torch.version.cuda or ""
print(f"torch={torch.__version__} built_for_cuda={cuda}")
if cuda.split(".")[0] != "12":
    print(f"FATAL: need a CUDA 12.x build (driver 570.x = CUDA 12.8), got {cuda!r}")
    sys.exit(1)

# A torch downgrade can break the simulator stack; catch it here, not on the GPU node.
import sapien
import mani_skill.envs  # noqa: F401
import mani_skill

print(f"sapien={sapien.__version__} mani_skill={mani_skill.__version__}")
PY

# Job 53283: the marker was written before verification, so a broken environment
# looked ready. Only mark ready once every check above has passed.
echo "$SPEC" > "$MARKER"
echo "[setup] marked ready: $SPEC"

echo "[setup] recording provenance"
uv pip freeze --python "$VENV/bin/python" > "$ROOT/outputs/pip_freeze.txt"
"$VENV/bin/python" - <<'PY' > "$ROOT/outputs/setup_versions.txt"
import importlib
for mod in ("mani_skill", "sapien", "torch", "gymnasium", "numpy"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}={getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{mod}=IMPORT_FAILED {exc!r}")
PY

cat "$ROOT/outputs/setup_versions.txt"
echo "[setup] done $(date -Is)"
