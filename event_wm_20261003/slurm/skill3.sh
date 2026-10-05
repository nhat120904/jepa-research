#!/usr/bin/env bash
#SBATCH --job-name=ew_skill3
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Skill v3 (CNN on raw pixels, end to end) on the round-2 events/planner.
#   PART=train sbatch --cpus-per-task=6 --mem=64G slurm/skill3.sh ;  PART=loop OUT=<train dir> sbatch slurm/skill3.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-puzzle-4x5-play-v0; CACHE=$RUN_ROOT/cache/$ENV
R2=${R2:-$RUN_ROOT/round2_56980}; S2=${S2:-$RUN_ROOT/skill2_56986/full/skill.pt}
OUT=${OUT:-$RUN_ROOT/skill3_${SLURM_JOB_ID}}
record_source "$OUT/src_${SLURM_JOB_ID}"
case ${PART:?PART} in
  train)
    "$TORCH_PY" "$PROJECT/scripts/train_skill3.py" --cache "$CACHE" --events "$R2/events" --skill2 "$S2" --release 3 --out "$OUT/full_rel3" ${SKILL_ARGS:-}
    "$TORCH_PY" "$PROJECT/scripts/train_skill3.py" --cache "$CACHE" --events "$R2/events" --skill2 "$S2" --release 3 --max-tau 20 --out "$OUT/tau20_rel3" ${SKILL_ARGS:-}
    ;;
  loop)
    IFS=";" read -ra LIST <<< "${ARMS:-full_rel3 oracle;tau20_rel3 oracle;full_rel3 learned;tau20_rel3 learned}"   # ARMS="skill high;skill high"
    for ARM in "${LIST[@]}"; do
      set -- $ARM
      "$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env "$ENV" --planner "$R2/planner/planner.pt" --skill "$OUT/$1/skill.pt" \
        --low skill --high "$2" --episodes "${EPISODES:-6}" --workers 14 --out "$OUT/loop_${1}_${2}${TAG:-}" ${LOOP_ARGS:-}
    done
    ;;
esac
