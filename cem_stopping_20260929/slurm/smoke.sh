#!/usr/bin/env bash
#SBATCH --job-name=cemstop_smoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
# Unit tests, upstream-equivalence/prefix check, exact replay, tree == live
# on two development roots per task. Also measures per-root runtime.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
RUN_DIR="$RUN_ROOT/smoke_${SLURM_JOB_ID}"
record_source "$RUN_DIR"
"$PY" "$PROJECT/scripts/run_checks.py" --out-dir "$RUN_DIR" "$@"
