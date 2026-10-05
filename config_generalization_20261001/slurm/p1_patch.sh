#!/usr/bin/env bash
#SBATCH --job-name=cg_p1patch
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=01:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/config_generalization/logs/%x_%j.out
# P1 with patch-token latents: train the patch predictor on the frozen encoder of BASE (train_wm.py
# checkpoint), then run p1_puzzle.py in patch mode.
#   ENV=visual-puzzle-4x5-play-v0 BASE=.../wm.pt STEPS=40000 VSTEPS=100000 TEP=1000 sbatch slurm/p1_patch.sh
#   PATCH=<patch_wm.pt> skips predictor training (re-evaluation only).
#   EXCLUDE="6 7 11 12" NEWBASE=60000: held-out region (all listed buttons ON) removed from training of a
#   freshly trained base WM/encoder, the patch predictor, the probes and the value.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001/slurm/env.sh
export PYTHONPATH="/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel:/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-pretraining:$PYTHONPATH"
ENV=${ENV:?ENV}; BASE=${BASE:-}
STEPS=${STEPS:-40000}; VSTEPS=${VSTEPS:-100000}; TEP=${TEP:-1000}
if [ "${SMOKE:-0}" = 1 ]; then STEPS=300; VSTEPS=300; TEP=40; fi
OUT=$RUN_ROOT/p1patch_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
EXCL=(); [ -n "${EXCLUDE:-}" ] && EXCL=(--exclude $EXCLUDE)
if [ -n "${NEWBASE:-}" ]; then   # retrain the encoder/base WM on the (filtered) data first
  "$TORCH_PY" "$PROJECT/scripts/train_wm.py" --data "$DATA" --env "$ENV" --max-steps "$NEWBASE" --out "$OUT/base" "${EXCL[@]}"
  BASE=$OUT/base/wm.pt
fi
if [ -z "${PATCH:-}" ]; then
  "$TORCH_PY" "$PROJECT/scripts/train_patch_wm.py" --data "$DATA" --env "$ENV" --base "$BASE" --steps "$STEPS" --out "$OUT" "${EXCL[@]}"
  PATCH=$OUT/patch_wm.pt
fi
"$TORCH_PY" "$PROJECT/scripts/p1_puzzle.py" --data "$DATA" --env "$ENV" --wm "$BASE" --patch-wm "$PATCH" \
  --train-episodes "$TEP" --value-steps "$VSTEPS" --out "$OUT/p1.json" "${EXCL[@]}"
