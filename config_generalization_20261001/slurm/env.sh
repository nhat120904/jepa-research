# Shared environment for config_generalization jobs (sourced by batch scripts).
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001
OGB=/mnt/data/nhatnc129/jepa/ogbench
OGB_PY=$OGB/venv/bin/python           # numpy / scipy / mujoco / ogbench (JAX stack for official baselines)
TORCH_PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python   # torch + stable-worldmodel stack
DATA=$OGB/data
RUN_ROOT=/mnt/data/nhatnc129/jepa/config_generalization
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/logs"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
record_source() {
  mkdir -p "$1"
  (cd "$PROJECT" && find scripts slurm -type f \( -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum) > "$1/SOURCE_SHA256SUMS"
}
