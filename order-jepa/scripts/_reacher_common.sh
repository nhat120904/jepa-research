# Shared environment for the official LeWM-Reacher Stage-A audit.
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/order-jepa"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
SWM_SOURCE="$REPO/diagnosis/external/stable-worldmodel"
REACHER_ROOT=${ORDER_REACHER_ROOT:-/mnt/data/nhatnc129/jepa/order_jepa/reacher_stage_a}
PROVENANCE="$REACHER_ROOT/provenance.json"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
