#!/usr/bin/env bash
#SBATCH --job-name=ew_code3
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Event code round 3: SFA, then ICA inside the N slowest directions (N in ICA_DIMS), keeping the
# components that are binary and flip rarely once binarised (label-free selection).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}
BASE=${BASE:-$CG_RUNS/p1_56657_visual-puzzle-4x5-play-v0/wm.pt}
OUT=$RUN_ROOT/code3_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/sfa_code.py" --cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --base "$BASE" \
  --ica-dims ${ICA_DIMS:-32 64 128 256} ${EXTRA:-} --out "$OUT"
