#!/usr/bin/env bash
#SBATCH --job-name=ew_code2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Second round of the event code: (a) linear SFA + ICA, (b) MLP head with fewer bits, longer
# pair gaps and a stronger slowness weight.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}
BASE=${BASE:-$CG_RUNS/p1_56657_visual-puzzle-4x5-play-v0/wm.pt}
OUT=$RUN_ROOT/code2_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/sfa_code.py" --cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --base "$BASE" --out "$OUT/sfa"
"$TORCH_PY" "$PROJECT/scripts/train_code.py" --cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --base "$BASE" \
  --bits 24 --gap 30 --w-slow 4 --out "$OUT/mlp_k24_g30_s4"
