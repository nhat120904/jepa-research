#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l0
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l0_%j.out
# LIBERO L0 (CPU): build a pinned venv, download the SmolVLA LIBERO checkpoint, run the OSMesa render probe.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
RUN="$ROOT/libero_l0_${SLURM_JOB_ID}"
VENV="$RUN/venv"
UV=/home/nhatnc129/.local/bin/uv
[[ ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/scripts/libero" "$PROJECT/docs" "$RUN/code/"
find "$RUN/code" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS"
export UV_CACHE_DIR=/mnt/data/nhatnc129/jepa/cache/uv
"$UV" venv --python 3.12 "$VENV"
# lerobot 0.6.1 + hf-libero + robosuite 1.4.0 is the combination reported working in lerobot#4614;
# MuJoCo pinned below 3.8.1 (breaks LIBERO init states there).
"$UV" pip install --python "$VENV/bin/python" "lerobot[libero,smolvla]==0.6.1" "mujoco==3.3.7"
"$UV" pip freeze --python "$VENV/bin/python" > "$RUN/pip_freeze.txt"
export HF_HOME="$RUN/hf_cache" HF_HUB_DISABLE_XET=1
"$VENV/bin/python" - <<PY
import json
from huggingface_hub import HfApi, snapshot_download
out = {}
for repo in ("HuggingFaceVLA/smolvla_libero", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    info = HfApi().model_info(repo)
    path = snapshot_download(repo, revision=info.sha)
    out[repo] = {"sha": info.sha, "path": path}
json.dump(out, open("$RUN/checkpoints.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PY
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH="$RUN/libero_config"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VENV/bin/python" "$RUN/code/libero/l0_probe.py" --run "$RUN" < /dev/null
