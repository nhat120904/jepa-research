#!/usr/bin/env bash
#SBATCH --job-name=cemstop_live_v2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
export CEMSTOP_REPO=/home/nhatnc129/nhat.nc/jepa-research
RUN="$RUN_ROOT/live_v2_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
cp -a "$PROJECT/cemstop" "$PROJECT/scripts" "$PROJECT/tests" "$PROJECT/docs" "$RUN/code/"
export PYTHONPATH="$RUN/code:$SWM_SOURCE"
"$PY" "$RUN/code/scripts/live_v2.py" --policies "${POLICIES:?set POLICIES}" \
  --data "$RUN_ROOT/dev_v1" --out "$RUN/results" --roots "${ROOTS:-100}" ${LIVE_EXTRA_ARGS:-}
