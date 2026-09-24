#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l3
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l3_%A_%a.out
# LIBERO L3 (docs/LIBERO_QUALIFICATION_PROTOCOL.md, Amendment 2): closed-loop P0 vs ORACLE8 (BDDL goal progress)
# on the 100 qualification roots (array task = LIBERO-Goal task id, init states 0-9). CPU only; both arms share
# the device. Goal-progress unit tests run first (Amendment 2: tested before any L3 rollout).
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
SETUP="$ROOT/libero_l0_54414"
VENV="$SETUP/venv"
RUN="$ROOT/libero_l3_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CODE" ]] || exit 2
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/cta_tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$SETUP/hf_cache" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 LIBERO_CONFIG_PATH="$SETUP/l1_libero_config"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$VENV/bin/python" -m unittest discover -s "$CODE/cta_tests" -p "test_goal_progress.py"
"$VENV/bin/python" "$CODE/scripts/libero/l3_oracle.py" --run "$RUN" --setup "$SETUP" \
  --tasks "$SLURM_ARRAY_TASK_ID" --inits 0 1 2 3 4 5 6 7 8 9 --device cpu < /dev/null
