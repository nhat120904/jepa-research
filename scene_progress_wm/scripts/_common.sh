# Shared environment for every Scene progress-WM job.
# Sourced, never executed on the login node.
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/scene_progress_wm"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
DATA_ROOT=/mnt/data/vhoangth2/datasets/ogbench_data
CACHE_ROOT=/mnt/data/nhatnc129/jepa/scene_progress_wm
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_data.py" "$PROJECT/scripts/_common.sh"
