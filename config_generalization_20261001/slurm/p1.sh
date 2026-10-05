#!/usr/bin/env bash
#SBATCH --job-name=cg_p1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/config_generalization/logs/%x_%j.out
# P1 puzzle: train the LeWM-style world model on one OGBench visual puzzle dataset, then evaluate
# event prediction and the frozen-latent GCIVL value by configuration novelty.
#   ENV=visual-puzzle-4x5-play-v0 STEPS=60000 VSTEPS=100000 sbatch slurm/p1.sh
#   SMOKE=1 -> 200 WM steps, 300 value steps. WM=<path to wm.pt> skips training (re-evaluation).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001/slurm/env.sh
export PYTHONPATH="/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel:/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-pretraining:$PYTHONPATH"
ENV=${ENV:?ENV}
STEPS=${STEPS:-60000}; VSTEPS=${VSTEPS:-100000}
if [ "${SMOKE:-0}" = 1 ]; then STEPS=200; VSTEPS=300; fi
OUT=$RUN_ROOT/p1_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "${WM:-}" ] || { "$TORCH_PY" "$PROJECT/scripts/train_wm.py" --data "$DATA" --env "$ENV" --max-steps "$STEPS" --out "$OUT"; WM=$OUT/wm.pt; }
case "$ENV" in
  *puzzle*) "$TORCH_PY" "$PROJECT/scripts/p1_puzzle.py" --data "$DATA" --env "$ENV" --wm "$WM" --value-steps "$VSTEPS" --out "$OUT/p1.json" ;;
  *cube-triple*) "$TORCH_PY" "$PROJECT/scripts/p1_cube.py" --data "$DATA" --env "$ENV" --wm "$WM" --slices 14 21 28 --value-steps "$VSTEPS" --out "$OUT/p1.json" ;;
  *) echo "no P1 evaluator for $ENV" >&2; exit 1 ;;
esac
