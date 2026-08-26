#!/usr/bin/env bash
#SBATCH --job-name=tdj_wrk
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_wrk_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
TDJ="$REPO/diagnosis/external/td-jepa"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export MKL_NUM_THREADS=1 WANDB_MODE=disabled
cd "$TDJ"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "TDJEPA_COMMIT=$(git rev-parse HEAD)"

# num_workers is a dataloader parallelism knob, but it is also a value in the
# official recipe (config/train/lewm.yaml: num_workers 2).  The fidelity gate is
# the one place a recipe deviation is unacceptable, so it is only allowed if it
# is PROVEN not to change the computation.  Run the identical 200 steps at
# num_workers 2 and 8 and compare the logged losses step by step: identical
# losses prove the knob is pure I/O and the speedup may be taken.
for NW in 2 8; do
  echo "=== num_workers=$NW start $(date -u +%FT%TZ) ==="
  OMP_NUM_THREADS=1 "$STAGE0_ROOT/.venv/bin/python" train.py --config-name=ogb_train \
    data=ogb variant=td_jepa seed=3072 \
    ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
    num_workers=$NW \
    trainer.max_epochs=1 trainer.limit_train_batches=200 \
    wandb.enabled=false planning_eval.enabled=false \
    output_model_name=workers/nw${NW}_200steps 2>&1 | grep -E "step [0-9]+/|loss" | tail -40
  echo "=== num_workers=$NW end   $(date -u +%FT%TZ) ==="
done
