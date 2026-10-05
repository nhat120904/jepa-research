#!/usr/bin/env bash
#SBATCH --job-name=ew_cubeA
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Cube direction A (label-free object state). Sub-directories of $OUT are named by EV / RD / PL / SK
# (defaults events / reader / planner / skill) so new versions never overwrite earlier ones.
#   PART=discover sbatch slurm/cube_a.sh                       -> $OUT/discover
#   PART=prep OUT=<dir> [PLAN=1] sbatch slurm/cube_a.sh        -> $OUT/$EV (moves, pseudo-labels), $OUT/$RD [, $OUT/$PL]
#   PART=train OUT=<dir> sbatch slurm/cube_a.sh                -> $OUT/$PL (pixel WM + affordance), $OUT/$SK
#   PART=skill OUT=<dir> sbatch slurm/cube_a.sh                -> $OUT/$SK only
#   PART=loop OUT=<dir> sbatch --cpus-per-task=16 --mem=96G slurm/cube_a.sh  -> $OUT/loop_reader${TAG}
#   PART=attrib OUT=<dir> sbatch --cpus-per-task=16 --mem=96G --time=00:50:00 slurm/cube_a.sh  -> PRIVILEGED attribution arms
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-cube-triple-play-v0; CACHE=$RUN_ROOT/cache/$ENV
OUT=${OUT:-$RUN_ROOT/cubeA_${SLURM_JOB_ID}}
EV=${EV:-events}; RD=${RD:-reader}; PL=${PL:-planner}; SK=${SK:-skill}
record_source "$OUT/src_${SLURM_JOB_ID}"
case ${PART:?PART} in
  discover)
    "$TORCH_PY" "$PROJECT/scripts/cube_discover.py" --cache "$CACHE" --kind triple --out "$OUT/discover" ${DISCOVER_ARGS:-}
    ;;
  prep)
    "$TORCH_PY" "$PROJECT/scripts/cube_events_px.py" --discover "$OUT/discover" --cache "$CACHE" \
      --ref-events "$RUN_ROOT/cubeD_57116/events" --out "$OUT/$EV"
    if [[ "${PLAN:-0}" == 1 ]]; then
      "$TORCH_PY" "$PROJECT/scripts/cube_planner.py" --events "$OUT/$EV" --out "$OUT/$PL"
    fi
    "$TORCH_PY" "$PROJECT/scripts/train_cube_reader.py" --cache "$CACHE" --discover "$OUT/discover" --events "$OUT/$EV" \
      --out "$OUT/$RD" ${READER_ARGS:-}
    ;;
  train)
    "$TORCH_PY" "$PROJECT/scripts/cube_planner.py" --events "$OUT/$EV" --out "$OUT/$PL"
    "$TORCH_PY" "$PROJECT/scripts/train_cube_skill.py" --cache "$CACHE" --events "$OUT/$EV" --out "$OUT/$SK" \
      --steps "${SKILL_STEPS:-20000}" ${SKILL_ARGS:-}
    ;;
  skill)
    if [[ "$EV" == events_refined && ! -f "$OUT/$EV/cube_events_train.npz" ]]; then   # reader-refined move timing
      "$TORCH_PY" "$PROJECT/scripts/cube_events_refine.py" --cache "$CACHE" --events "$OUT/events_cov2" \
        --reader "$OUT/reader_cov2/cube_reader.pt" --ref-events "$RUN_ROOT/cubeD_57116/events" --discover "$OUT/discover" --out "$OUT/$EV"
    fi
    if [[ "$EV" == events_qpos2px ]]; then   # PRIVILEGED attribution: qpos segments with pixel targets
      "$TORCH_PY" "$PROJECT/scripts/cube_events_qpos2px.py" --qpos-events "$RUN_ROOT/cubeD_57116/events" \
        --project "$OUT/projection/projection.npz" --px-events "$OUT/events_cov2" --out "$OUT/$EV"
    fi
    "$TORCH_PY" "$PROJECT/scripts/train_cube_skill.py" --cache "$CACHE" --events "$OUT/$EV" --out "$OUT/$SK" \
      --steps "${SKILL_STEPS:-20000}" ${SKILL_ARGS:-}
    ;;
  attrib)
    # PRIVILEGED attribution (matched seeds): (1) privileged high level (D planner + true state) + label-free skill;
    # (2) label-free pixel planner + skill with perfect perception (true state projected to pixels, privileged move ends)
    [[ -f "$OUT/projection/projection.npz" ]] || "$TORCH_PY" "$PROJECT/scripts/cube_projection.py" --cache "$CACHE" --discover "$OUT/discover" --out "$OUT/projection"
    if [[ "${ATTRIB_ARMS:-both}" != oracle ]]; then
      for SKI in ${SKS:-$SK}; do      # SKS: several skill dirs evaluated under the privileged high level
        "$TORCH_PY" "$PROJECT/scripts/cube_closed_loop.py" --perception privileged --planner "$RUN_ROOT/cubeD_57116/planner/cube_planner.pt" \
          --skill "$OUT/$SKI/${SKILL:-cube_skill_best.pt}" --project "$OUT/projection/projection.npz" --events "$OUT/$EV" \
          --cache "$CACHE" --episodes "${EPISODES:-6}" --workers 14 --out "$OUT/attrib_privhigh_${SKI}${TAG:-}"
      done
    fi
    [[ "${ATTRIB_ARMS:-both}" == priv ]] || "$TORCH_PY" "$PROJECT/scripts/cube_closed_loop.py" --perception oracle_px --planner "$OUT/$PL/cube_planner.pt" \
      --skill "$OUT/$SK/${SKILL:-cube_skill_best.pt}" --project "$OUT/projection/projection.npz" --events "$OUT/$EV" \
      --cache "$CACHE" --episodes "${EPISODES:-6}" --workers 14 --out "$OUT/attrib_oraclepx_${PL}_${SK}"
    ;;
  loop)
    "$TORCH_PY" "$PROJECT/scripts/cube_closed_loop.py" --perception reader --reader "$OUT/$RD/cube_reader.pt" \
      --planner "$OUT/$PL/cube_planner.pt" --skill "$OUT/$SK/${SKILL:-cube_skill_best.pt}" --events "$OUT/$EV" \
      --discover "$OUT/discover" --cache "$CACHE" --episodes "${EPISODES:-6}" --episode-start "${EP_START:-0}" --workers 14 \
      --out "$OUT/loop_reader${TAG:-}" ${LOOP_ARGS:-}
    ;;
esac
