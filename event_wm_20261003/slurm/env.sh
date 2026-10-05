# Shared environment for event_wm jobs (sourced by batch scripts).
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003
CG=/home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001
OGB=/mnt/data/nhatnc129/jepa/ogbench
OGB_PY=$OGB/venv/bin/python           # numpy / mujoco / ogbench envs
TORCH_PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python   # torch + stable-worldmodel stack
DATA=$OGB/data
RUN_ROOT=/mnt/data/nhatnc129/jepa/event_wm
CG_RUNS=/mnt/data/nhatnc129/jepa/config_generalization
SWM=/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external
# The cluster has no GPU GL stack: MuJoCo renders through CPU OSMesa.
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/scripts:$CG/scripts:$SWM/stable-worldmodel:$SWM/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/logs"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
record_source() {
  mkdir -p "$1"
  (cd "$PROJECT" && find scripts slurm -type f \( -name '*.py' -o -name '*.sh' \) -print0 | sort -z | xargs -0 sha256sum) > "$1/SOURCE_SHA256SUMS"
}
