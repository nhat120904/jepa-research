#!/usr/bin/env bash
#SBATCH --job-name=lscope_scrub_data
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/scrub_data_check_%j.out

# CPU only: recorded-state playback with the native Scrub monitor. No GPU, policy or rendering.
set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
VENV_PATH="$STAGE_ROOT/venv"
ROBOCASA_SRC="$STAGE_ROOT/src/robocasa365"
ROBOSUITE_SRC="$STAGE_ROOT/src/robosuite"
OUT=/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/scrub_data_check_$SLURM_JOB_ID

mkdir -p "$OUT/code" /mnt/data/nhatnc129/jepa/latent_scope_data
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/check_scrub_demo_data.py" \
   "$PROJECT_ROOT/latent_scope_20260909/configs/scrub_data_check.json" "$OUT/code/"
sha256sum "$OUT"/code/* > "$OUT/code/SHA256SUMS"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$ROBOCASA_SRC:$ROBOSUITE_SRC:${PYTHONPATH:-}"

"$VENV_PATH/bin/python" "$OUT/code/check_scrub_demo_data.py" \
    --config "$OUT/code/scrub_data_check.json" \
    --output-dir "$OUT"
