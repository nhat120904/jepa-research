#!/usr/bin/env bash
#SBATCH --job-name=ti_ogb_v2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/ogb_%x_%j.out
# CTA v2 on OGBench visual-cube-single (scripts/ogbench/ogb_cta_train_v2.py). Modes:
#   smoke   unit tests + v2 training on a small slice (few steps): code path only, not a result
#   train   full v2 training + offline ladder. SLURM_ARRAY_TASK_ID selects the goal setting:
#           0 = all five official goals (in-distribution), 1 = goals 4,5 held out (never trained on)
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
DATA=/mnt/data/nhatnc129/jepa/ogbench/enc/visual-cube-single-play-v0
OUT=/mnt/data/nhatnc129/jepa/ogbench/cta_v2
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch PYTHONUNBUFFERED=1
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID TASK=${SLURM_ARRAY_TASK_ID:-} MODE=$MODE $(date -u +%FT%TZ)"
# snapshot of the exact sources this job runs
SNAP="$OUT/code_${SLURM_JOB_ID}"
mkdir -p "$SNAP"
cp "$PROJ"/scripts/ogbench/ogb_cta_train_v2.py "$PROJ"/scripts/ogbench/ogb_cta_train.py "$PROJ"/scripts/ogbench/ogb_encode_play.py "$PROJ"/scripts/ogbench/ogb_cta_train_play.py "$PROJ"/scripts/ogbench/ogb_cta_eval.py "$SNAP"/
cp "$PROJ"/ti_wm/cta.py "$PROJ"/ti_wm/cta_parallel.py "$PROJ"/ti_wm/cta_ogb.py "$PROJ"/ti_wm/cta_eval.py "$SNAP"/
sha256sum "$SNAP"/* > "$SNAP/SHA256SUMS"
cd "$PROJ"
"$PY" -m unittest discover -s cta_tests -p "test_cta_ogb*.py"
case "$MODE" in
  smoke)
    "$PY" scripts/ogbench/ogb_cta_train_v2.py --run "$OUT/smoke_${SLURM_JOB_ID}" --data "$DATA" --limit 2000 \
      --steps1 60 --steps2 40 --eval-every 30 --heldout-goals 4,5
    ;;
  train)
    case "${SLURM_ARRAY_TASK_ID:-0}" in
      0) ARGS=""; NAME=iid ;;
      1) ARGS="--heldout-goals 4,5"; NAME=lto45 ;;
      2) ARGS="--levels 8,8,8,5,5 --steps2 3000"; NAME=iid_lv16 ;;
      3) ARGS="--nested 0.5 --steps2 3000"; NAME=iid_nested ;;
      *) echo "unknown task" >&2; exit 2 ;;
    esac
    "$PY" scripts/ogbench/ogb_cta_train_v2.py --run "$OUT/${NAME}_s${SEED:-0}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
      --data "$DATA" --in-ram --seed "${SEED:-0}" $ARGS ${EXTRA_ARGS:-}
    ;;
  smoke_play)
    "$PY" scripts/ogbench/ogb_cta_train_play.py --run "$OUT/smoke_play_${SLURM_JOB_ID}" \
      --play /mnt/data/nhatnc129/jepa/ogbench/enc_play/visual-cube-single-play-v0 --banks "$DATA" --max-frames 30000 \
      --in-ram --steps1 40 --steps2 30 --eval-every 20 --heldout-goals 4,5
    ;;
  train_play)
    case "${SLURM_ARRAY_TASK_ID:-0}" in
      0) ARGS=""; NAME=play_iid ;;
      1) ARGS="--heldout-goals 4,5"; NAME=play_lto45 ;;
      *) echo "unknown task" >&2; exit 2 ;;
    esac
    "$PY" scripts/ogbench/ogb_cta_train_play.py --run "$OUT/${NAME}_s${SEED:-0}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
      --play /mnt/data/nhatnc129/jepa/ogbench/enc_play/visual-cube-single-play-v0 --banks "$DATA" --in-ram \
      --seed "${SEED:-0}" $ARGS ${EXTRA_ARGS:-}
    ;;
  eval)
    # closed loop (scripts/ogbench/ogb_cta_eval.py): same frozen policy and seeded banks as P0 (P0 is identical across
    # runs, so it is not re-run); CKPT = a v2/play checkpoint; dev episodes by default
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    "$PY" scripts/ogbench/ogb_cta_eval.py --run "${EVAL_RUN:?EVAL_RUN}" --env visual-cube-single-play-v0 \
      --policy /mnt/data/nhatnc129/jepa/ogbench/gcfbc/visual-cube-single-play-v0_55415/policy_500000.pt \
      --ckpt "${CKPT:?CKPT}" --pca "$DATA/pca.pt" --arms "${ARMS:-CTA,DIRECT,ENDPOINT,FRAME}" \
      --tasks "${TASKS:-1,2,3,4,5}" --first "${FIRST:-0}" --count "${COUNT:-20}"
    ;;
  encode_play)
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    "$PY" scripts/ogbench/ogb_encode_play.py --out /mnt/data/nhatnc129/jepa/ogbench/enc_play/visual-cube-single-play-v0 ${EXTRA_ARGS:-}
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "DONE $(date -u +%FT%TZ)"
