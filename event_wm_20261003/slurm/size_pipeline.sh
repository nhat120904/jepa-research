#!/usr/bin/env bash
#SBATCH --job-name=ew_size
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# The full 4x5 recipe for another grid size, in three chained parts (all label-free; privileged
# button states only score each stage):
#   PART=code  SFA + band-wise ICA (N in ICA_DIMS; the N with the most selected binary components wins),
#              events, segment-majority refinement, events again, evaluation task frames
#   PART=plan  event WM + cost-to-go (width 2048 + xor) + offline checks, skill v3 (tau20, release 3)
#   PART=loop  closed loop: learned high level and PRIVILEGED oracle plan, both with skill v3
#   ENV=visual-puzzle-4x6-play-v0 BASE=<wm.pt> OUT=<dir> PART=code sbatch slurm/size_pipeline.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:?ENV}; SIZE=$(echo "$ENV" | cut -d- -f3); CACHE=$RUN_ROOT/cache/$ENV
OUT=${OUT:?OUT}
record_source "$OUT/src_${SLURM_JOB_ID}"
case ${PART:?PART} in
  code)
    BASE=${BASE:?BASE}
    "$TORCH_PY" "$PROJECT/scripts/sfa_code.py" --cache "$CACHE" --env "$ENV" --base "$BASE" --bands \
      --ica-dims ${ICA_DIMS:-64 96} --out "$OUT/sfa"
    N=$("$TORCH_PY" -c "
import json,sys; s=json.load(open('$OUT/sfa/summary.json'))
print(sorted(s, key=lambda n: (-s[n]['selected'], int(n)))[0])")
    echo "chosen ICA dims N=$N (most selected binary components)"
    "$TORCH_PY" "$PROJECT/scripts/build_events.py" --cache "$CACHE" --env "$ENV" --code "$OUT/sfa/N$N/code.pt" --out "$OUT/events_ica"
    "$TORCH_PY" "$PROJECT/scripts/refine_code.py" --cache "$CACHE" --env "$ENV" --code "$OUT/sfa/N$N/code.pt" \
      --events "$OUT/events_ica" --labels segment --margin 3 --out "$OUT/refine"
    "$TORCH_PY" "$PROJECT/scripts/build_events.py" --cache "$CACHE" --env "$ENV" --code "$OUT/refine/code.pt" --out "$OUT/events"
    "$TORCH_PY" "$PROJECT/scripts/render_check.py" --size "$SIZE" --cache "$CACHE" --out "$OUT/tasks"
    ;;
  plan)
    "$TORCH_PY" "$PROJECT/scripts/train_planner.py" --events "$OUT/events" --cache "$CACHE" --env "$ENV" \
      --code "$OUT/refine/code.pt" --tasks "$OUT/tasks/tasks_${SIZE}.npz" --h-xor --h-steps ${HSTEPS:-150000} --out "$OUT/planner"
    # skill v3 reuses a v2 checkpoint only for its target maps; compute them with a short v2 run
    "$TORCH_PY" "$PROJECT/scripts/train_skill2.py" --cache "$CACHE" --events "$OUT/events" --code "$OUT/refine/code.pt" \
      --steps 200 --out "$OUT/tmaps"
    "$TORCH_PY" "$PROJECT/scripts/train_skill3.py" --cache "$CACHE" --events "$OUT/events" --skill2 "$OUT/tmaps/skill.pt" \
      --release 3 --max-tau 20 --out "$OUT/skill3"
    ;;
  loop)
    IFS=";" read -ra LIST <<< "${ARMS:-learned;oracle}"
    for HIGH in "${LIST[@]}"; do
      "$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env "$ENV" --planner "$OUT/planner/planner.pt" --skill "$OUT/skill3/skill.pt" \
        --low skill --high "$HIGH" --episodes "${EPISODES:-6}" --workers 14 --out "$OUT/loop_${HIGH}${TAG:-}" ${LOOP_ARGS:-}
    done
    ;;
esac
