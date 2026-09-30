#!/usr/bin/env bash
#SBATCH --job-name=ar_cube_hard
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response_cube/logs/%x_%j.out
set -euo pipefail

PROJECT=/home/nhatnc129/nhat.nc/jepa-research/action_response_cube_20260928
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export STABLEWM_HOME=/mnt/data/nhatnc129/jepa/lewm_stage0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=8
export PYTHONPATH=/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel:${PYTHONPATH:-}
PARENT=/mnt/data/nhatnc129/jepa/action_response_cube/run_55413_20260928T023327Z
OUT=/mnt/data/nhatnc129/jepa/action_response_cube/hard_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
printf '%s\n' "$OUT" > /mnt/data/nhatnc129/jepa/action_response_cube/job_${SLURM_JOB_ID}.path
sha256sum "$PROJECT/pipeline.py" "$PROJECT/eval_hard_start.py" "$PROJECT/slurm_hard_start.sh" > "$OUT/SOURCE_SHA256SUMS"
"$PY" "$PROJECT/eval_hard_start.py" \
  --parent-run "$PARENT" --out-dir "$OUT" \
  --roots "${ROOTS:-50}" --seed "${SEED:-20260929}"
