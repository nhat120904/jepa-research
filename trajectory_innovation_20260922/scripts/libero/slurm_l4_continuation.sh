#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l4
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l4_%x_%A_%a.out
# LIBERO L4 (docs/CTA_LIBERO_CONTINUATION_PROTOCOL.md): continuation-success labels for K=8 SmolVLA chunks.
#   collect   (array 0-19)  root = (task idx/2, init idx%2); CPU only, SmolVLA on CPU as at L2/L3
#   aggregate (no array)    RUN=<collect run dir>; held-out selector gains, bootstrap over roots
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
SETUP="$ROOT/libero_l0_54414"
VENV="$SETUP/venv"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$SETUP/hf_cache" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 LIBERO_CONFIG_PATH="$SETUP/l1_libero_config"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID:-NA} $(date -u +%FT%TZ)"
case "$MODE" in
  collect)
    I=${SLURM_ARRAY_TASK_ID:?array task}
    RUN="$ROOT/libero_l4_${SLURM_ARRAY_JOB_ID}"
    CODE="$RUN/code_${I}"
    [[ ! -e "$CODE" ]] || exit 2
    mkdir -p "$CODE"
    cp -r "$PROJECT/ti_wm" "$PROJECT/cta_tests" "$PROJECT/scripts" "$CODE/"
    find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS_${I}"
    export PYTHONPATH="$CODE"
    "$VENV/bin/python" -m unittest discover -s "$CODE/cta_tests" -p "test_libero_continuation.py"
    "$VENV/bin/python" "$CODE/scripts/libero/l4_continuation.py" collect --run "$RUN" --setup "$SETUP" \
      --tasks $((I / 2)) --inits $((I % 2)) --device cpu < /dev/null
    ;;
  aggregate)
    export PYTHONPATH="$PROJECT"
    "$VENV/bin/python" "$PROJECT/scripts/libero/l4_continuation.py" aggregate --run "${RUN:?RUN}"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
