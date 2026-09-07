#!/usr/bin/env bash
#SBATCH --job-name=tdj_smoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_smoke_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
TDJ="$REPO/diagnosis/external/td-jepa"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
cd "$TDJ"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "TDJEPA_COMMIT=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# One epoch only, to time the official recipe and prove the path runs.
# planning_eval is disabled here: it would add a full CEM eval per epoch and
# this smoke measures training throughput, not planning.
echo "--- train start $(date -u +%FT%TZ) ---"
# ~data.dataset.rdcc_{nbytes,w0}: this checkout of stable-worldmodel
# (9a66d7d, 2026-08-10) is NEWER than TD-JEPA (2026-07-29) and no longer takes
# these as kwargs -- it hardcodes rdcc_nbytes=256MiB, the identical value the
# config asks for, leaving only rdcc_w0 (1.0 vs h5py's 0.75).  Both are h5py
# chunk-CACHE knobs: they change read speed, never the bytes returned.  Removing
# them via Hydra keeps the vendored repo pristine and, more importantly, keeps
# TD-JEPA, our LeWM control and the Phase-1 arena on ONE stable-worldmodel.
# Giving TD-JEPA a different upstream version would put the arms in different
# environments, which is a far worse confound than a cache policy.
"$STAGE0_ROOT/.venv/bin/python" train.py --config-name=ogb_train \
  data=ogb variant=td_jepa seed=3072 \
  ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
  trainer.max_epochs=1 wandb.enabled=false planning_eval.enabled=false \
  output_model_name=smoke/td_jepa/seed_3072_1ep
echo "--- train end $(date -u +%FT%TZ) ---"
ls -la "$STABLEWM_HOME/checkpoints/smoke/td_jepa/seed_3072_1ep/" || true
