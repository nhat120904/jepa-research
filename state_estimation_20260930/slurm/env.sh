# Shared environment for state_estimation jobs (sourced by the batch scripts).
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/state_estimation_20260930
SWM_SOURCE=/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN_ROOT=/mnt/data/nhatnc129/jepa/state_estimation
export STABLEWM_HOME=/mnt/data/nhatnc129/jepa/lewm_stage0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$PROJECT:$SWM_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/logs"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

record_source() {
  local dir=$1
  mkdir -p "$dir"
  (cd "$PROJECT" && find se scripts slurm -type f \( -name '*.py' -o -name '*.sh' \) \
     -print0 | sort -z | xargs -0 sha256sum) > "$dir/SOURCE_SHA256SUMS"
  git -C "$SWM_SOURCE" rev-parse HEAD > "$dir/STABLE_WORLDMODEL_COMMIT"
}
