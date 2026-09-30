#!/usr/bin/env bash
#SBATCH --job-name=ti_libsafe
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libsafe_%x_%j.out
# LIBERO-Safety arena for CTA (docs/CTA_LIBSAFE_PROTOCOL.md).
#   setup     (main, CPU)  openpi venv from its uv.lock + LIBERO-Safety (pinned commits), pi0.5 checkpoint (pinned HF
#                          revision), benchmark assets, PaliGemma tokenizer; then the CPU probe (scripts/libsafe/probe.py)
#   probe     (main, CPU)  the CPU probe alone (after setup)
#   headroom  (mig GPU)    P0 / ORACLE8 with the frozen pi0.5, $WORKERS processes sharing one MIG slice, then aggregate
#     sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --time=... slurm_libsafe.sh headroom
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
BASE=/mnt/data/nhatnc129/jepa/libero_safety
PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
REPO="$BASE/LIBERO-Safety"; OPENPI="$BASE/openpi"; VENV="$BASE/venv"; CKPT="$BASE/ckpt/pi05_libero_safety"
LIBSAFE_COMMIT=19ec8df23eedfbb9265bafd3e56495fcebfcfcd0
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
CKPT_REV=e66d84bfec17770384550593e39975f7e21e3463
UV=/home/nhatnc129/.local/bin/uv
export UV_CACHE_DIR=/mnt/data/nhatnc129/jepa/cache/uv HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME="$BASE/openpi_cache" PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
# LIBERO's top-level `libero/` has no __init__.py, which the editable install's finder cannot map (openpi's LIBERO
# instructions put the checkout on PYTHONPATH for the same reason)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$VENV/bin/python"
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID MODE=$MODE $(date -u +%FT%TZ)"
case "$MODE" in
  setup)
    git -C "$REPO" checkout --quiet "$LIBSAFE_COMMIT"; git -C "$OPENPI" checkout --quiet "$OPENPI_COMMIT"
    git -C "$OPENPI" submodule update --init --recursive --quiet || true
    # openpi's locked environment (python 3.11, jax 0.5.3, flax 0.10.2, torch 2.7.1), created outside the repo
    ( cd "$OPENPI" && GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen --python 3.11 )
    # LIBERO-Safety on top without touching openpi's pins: the package and its robosuite fork without dependencies,
    # then the few runtime imports they need; MuJoCo 3.3.7 (LIBERO init states break at >= 3.8.1; mj_getState needed)
    "$UV" pip install --python "$PY" --no-deps -e "$REPO" -e "$REPO/third_party/robosuite-1.4"
    "$UV" pip install --python "$PY" "mujoco==3.3.7" "bddl==1.0.1" easydict future termcolor h5py scikit-image numba \
      pynput "opencv-python-headless<4.11" pyyaml "gym==0.25.2" PyOpenGL
    RS="$REPO/third_party/robosuite-1.4/robosuite"
    # robosuite logs to a hardcoded /tmp/robosuite.log owned by another user on some nodes (see scripts/libero):
    # its documented override file turns file logging off
    [[ -e "$RS/macros_private.py" ]] || sed -e 's/^FILE_LOGGING_LEVEL = .*/FILE_LOGGING_LEVEL = None/' \
      -e 's/^MUJOCO_GPU_RENDERING = .*/MUJOCO_GPU_RENDERING = False/' "$RS/macros.py" > "$RS/macros_private.py"
    "$UV" pip freeze --python "$PY" > "$BASE/pip_freeze.txt"
    "$PY" - <<PY
import json, os, zipfile
from huggingface_hub import hf_hub_download, snapshot_download
out = {}
out["checkpoint"] = snapshot_download("LIBERO-Safety/pi05_libero_safety", revision="$CKPT_REV", local_dir="$CKPT")
root = "$REPO/libero/libero"
if not os.path.exists(os.path.join(root, "assets_ok")):
    z = hf_hub_download("LIBERO-Safety/libero_safety_assets", "assets.zip", repo_type="dataset", local_dir="$BASE/assets_zip")
    with zipfile.ZipFile(z) as f:
        f.extractall(root)
    open(os.path.join(root, "assets_ok"), "w").write(z)
out["assets"] = sorted(os.listdir(os.path.join(root, "assets")))[:50]
from openpi.shared import download
out["tokenizer"] = str(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
json.dump(out, open("$BASE/setup_report.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PY
    ;&
  probe)
    # dependencies added after the first setup (idempotent)
    "$UV" pip install --python "$PY" "gym==0.25.2" PyOpenGL
    RUN="$BASE/probe_$SLURM_JOB_ID"; mkdir -p "$RUN"
    OMP_NUM_THREADS=1 "$PY" "$PROJ/scripts/libsafe/probe.py" --run "$RUN" --repo "$REPO" < /dev/null
    ;;
  headroom)
    RUN="$BASE/headroom_${SLURM_JOB_ID}"; mkdir -p "$RUN/code/scripts"
    # keep the project layout: headroom.py imports ti_wm from parents[2] (55634 failed with code/libsafe)
    cp -r "$PROJ/scripts/libsafe" "$RUN/code/scripts/"; cp -r "$PROJ/ti_wm" "$RUN/code/"
    # jax reads nvidia.cuda_nvcc.__file__ (None for uv's namespace package): point it at the pip CUDA toolkit
    export CUDA_ROOT="$VENV/lib/python3.11/site-packages/nvidia/cuda_nvcc"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    W=${WORKERS:-3}
    PIDS=()
    for i in $(seq 0 $((W - 1))); do
      "$PY" "$RUN/code/scripts/libsafe/headroom.py" --run "$RUN" --repo "$REPO" --checkpoint "$CKPT" --shard "$i" \
        --shards "$W" ${HEADROOM_ARGS:-} > "$RUN/worker$i.log" 2>&1 < /dev/null &
      PIDS+=($!)
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done   # a failed worker fails the job (smoke gates the run)
    tail -n 3 "$RUN"/worker*.log
    "$PY" "$RUN/code/scripts/libsafe/headroom.py" --run "$RUN" --aggregate
    [[ $FAIL == 0 ]] || { echo "a headroom worker failed" >&2; exit 1; }
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
