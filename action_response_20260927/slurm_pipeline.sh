#!/usr/bin/env bash
#SBATCH --job-name=ar_lewm_pusht
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response/logs/%x_%j.out
set -euo pipefail

PROJECT=/home/nhatnc129/nhat.nc/jepa-research/action_response_20260927
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export STABLEWM_HOME=/mnt/data/nhatnc129/jepa/lewm_stage0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=8
export PYTHONPATH=/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel:${PYTHONPATH:-}
RUN_ROOT=${RUN_ROOT:-/mnt/data/nhatnc129/jepa/action_response}
mkdir -p "$RUN_ROOT"
RUN_DIR="$RUN_ROOT/run_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
sha256sum "$PROJECT/pipeline.py" "$PROJECT/slurm_pipeline.sh" > "$RUN_DIR/SOURCE_SHA256SUMS"
git -C /home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel rev-parse HEAD > "$RUN_DIR/STABLE_WORLDMODEL_COMMIT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/job_${SLURM_JOB_ID}.path"
EXTRA=()
if [[ -n "${BRANCH_CACHE:-}" ]]; then
  EXTRA+=(--branch-cache "$BRANCH_CACHE")
fi
"$PY" "$PROJECT/pipeline.py" \
  --run-dir "$RUN_DIR" \
  --pairs "${PAIRS:-96}" \
  --updates "${UPDATES:-160}" \
  --eval-roots "${EVAL_ROOTS:-4}" \
  --seed "${SEED:-27}" \
  --lr "${LR:-2e-6}" \
  --response-target "${RESPONSE_TARGET:-all3}" \
  "${EXTRA[@]}"
