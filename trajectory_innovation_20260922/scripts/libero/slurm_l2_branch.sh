#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l2
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l2_%A_%a.out
# LIBERO L2 (docs/LIBERO_QUALIFICATION_PROTOCOL.md + Amendment 2): branching fidelity (hard requirement) and
# candidate diversity (reported). CPU only: SmolVLA runs on CPU here, which is enough for fidelity/diversity;
# P0 success on GPU is measured at L3. Task pair (2i, 2i+1), init states 0-1 -> 20 roots in total.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
SETUP="$ROOT/libero_l0_54414"
VENV="$SETUP/venv"
RUN="$ROOT/libero_l2_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CODE" ]] || exit 2
mkdir -p "$CODE"
cp -r "$PROJECT/ti_wm" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="$CODE" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$SETUP/hf_cache" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 LIBERO_CONFIG_PATH="$SETUP/l1_libero_config"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
T=$SLURM_ARRAY_TASK_ID
"$VENV/bin/python" "$CODE/scripts/libero/l2_branch_check.py" --run "$RUN" --setup "$SETUP" \
  --tasks $((2 * T)) $((2 * T + 1)) --inits 0 1 --device cpu < /dev/null
