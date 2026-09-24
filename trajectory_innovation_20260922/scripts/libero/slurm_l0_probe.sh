#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l0p
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l0p_%j.out
# LIBERO L0 probe rerun on the venv built by job 54414.
# Fix: robosuite 1.4.0 logs to the hardcoded /tmp/robosuite.log (owned by another user on the node).
# We add robosuite/macros_private.py (robosuite's documented override) with file logging off and
# MUJOCO_GPU_RENDERING=False (CPU OSMesa). Nothing else in macros changes (IMAGE_CONVENTION stays "opengl").
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
SETUP="$ROOT/libero_l0_54414"
VENV="$SETUP/venv"
RUN="$ROOT/libero_l0p_${SLURM_JOB_ID}"
[[ ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/scripts/libero" "$PROJECT/docs" "$RUN/code/"
find "$RUN/code" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS"
RS=$("$VENV/bin/python" -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('robosuite').origin))")
if [[ ! -e "$RS/macros_private.py" ]]; then
  sed -e 's/^FILE_LOGGING_LEVEL = .*/FILE_LOGGING_LEVEL = None/' -e 's/^MUJOCO_GPU_RENDERING = .*/MUJOCO_GPU_RENDERING = False/' "$RS/macros.py" > "$RS/macros_private.py"
fi
diff "$RS/macros.py" "$RS/macros_private.py" > "$RUN/macros_private.diff" || true
cat "$RUN/macros_private.diff"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH="$RUN/libero_config"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VENV/bin/python" "$RUN/code/libero/l0_probe.py" --run "$RUN" < /dev/null
