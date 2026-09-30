#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_budget
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_budget_%x_%j.out
# Budget and paired-ladder analyses of the PushT v2 closed loop (55666).
#   ladder (CPU):  sbatch --partition=main --cpus-per-task=2 --mem=8G --time=00:20:00 slurm_cta_budget.sh ladder
#   budget (GPU):  sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 slurm_cta_budget.sh budget
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
MODE=${1:?ladder/budget}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
RUN="$TI/cta_${MODE}_$SLURM_JOB_ID"
CODE="$RUN/code"
[[ ! -e "$CODE" ]] || { echo "Existing code snapshot: $CODE" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG=cta_budget
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
S="$CODE/scripts"
case "$MODE" in
  ladder)
    "$PY" "$S/cta_paired_ladder.py" --closed-run "$TI/cta_v2cl_closed_55666" --out "$RUN"
    ;;
  direct)      # parameter-matched direct scorer (GPU): sbatch --partition=mig --gres=... --time=03:00:00 ... direct
    "$PY" "$S/cta_direct_matched.py" --run "$RUN/smoke" --v2 "$ROOT/cta_v2_train_55616/train_s0" --layers "${LAYERS:-14}" --smoke
    "$PY" "$S/cta_direct_matched.py" --run "$RUN/train" --v2 "$ROOT/cta_v2_train_55616/train_s0" --layers "${LAYERS:-14}"
    ;;
  ladder_rerun)  # CPU: $2 = closed job id of a P0-only rerun of 55666's roots
    "$PY" "$S/cta_paired_ladder.py" --closed-run "$TI/cta_v2cl_closed_${2:?closed job}" --out "$RUN" \
      --reference-run "$TI/cta_v2cl_closed_55666" \
      --pairs "CTAV2-DIRV2L,CTAV2-DIRV2,DIRV2L-DIRV2,CTAV2-ENDV2,ENDV2-DIRV2L,FULLV2-DIRV2L,FULLV2-CTAV2"
    ;;
  budget)
    "$PY" "$S/cta_budget.py" --out "$RUN" --parent "$ROOT/cta_geometry_e2e_55018/train" \
      --v2 "$ROOT/cta_v2_train_55616/train_s0" --r4 "$ROOT/cta_plus_train_55149/train" \
      --dinowm /mnt/data/nhatnc129/jepa/dino_wm_official/outputs/pusht --prep "$PREP" --smoke "$SMOKE"
    ;;
  *) echo "Unknown mode $MODE" >&2; exit 2 ;;
esac
echo "DONE $(date -u +%FT%TZ)"
