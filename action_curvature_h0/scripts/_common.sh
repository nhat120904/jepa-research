# Shared environment for every Action-Space Curvature Mismatch job.
# Sourced, never executed on the login node.
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
MANIFEST="$REPO/physical_search_distillation/outputs/h0/manifest.json"
POPULATIONS="$REPO/physical_search_distillation/outputs/h0/populations"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/measure_curvature.py" \
          "$PROJECT/scripts/aggregate.py" "$PROJECT/scripts/_common.sh"
