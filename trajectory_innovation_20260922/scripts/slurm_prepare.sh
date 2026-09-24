#!/usr/bin/env bash
#SBATCH --job-name=ti_policy_prepare
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/prepare_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
UV=/home/nhatnc129/.local/bin/uv
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
RUN="$ROOT/prepare_${SLURM_JOB_ID}"
[[ ! -e "$RUN" ]] || exit 2
mkdir -p "$RUN/code"
cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/scripts" "$RUN/code/"
cp "$PROJECT/IMPLEMENTATION_PLAN.md" "$PROJECT/EXPERIMENT_CONTRACT.md" "$RUN/code/"
find "$RUN/code" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export HF_HOME="$RUN/hf_cache" HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DISABLE_XET=1
git init -q "$RUN/upstream"
git -C "$RUN/upstream" remote add origin https://github.com/huggingface/lerobot.git
git -C "$RUN/upstream" fetch --depth 1 origin 3c0a209f9fac4d2a57617e686a7f2a2309144ba2
git -C "$RUN/upstream" checkout --detach FETCH_HEAD
# Isolated overlay: do not install into the shared base venv; reuse its torch wheels.
"$UV" pip install --python "$PY" --target "$RUN/deps" \
  'numpy==1.26.4' 'gymnasium==0.29.1' 'gym-pusht==0.1.5' \
  'pymunk==6.6.0' 'pygame==2.6.1' 'opencv-python==4.10.0.84' \
  'diffusers==0.32.2' 'draccus==0.10.0' 'huggingface-hub==0.36.2' \
  'safetensors==0.8.0' 'einops==0.8.2'
export PYTHONPATH="$RUN/code:$RUN/deps:$RUN/upstream"
"$PY" -m unittest discover -s "$RUN/code/tests" -v
"$PY" "$RUN/code/scripts/prepare_runtime.py" --run "$RUN"
