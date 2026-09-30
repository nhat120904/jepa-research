#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_v2cl
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_v2cl_%x_%A_%a.out
# PushT closed loop for CTA v2 (scripts/cta_diag_onpolicy.py --mode closed; lockstep, all candidates simulated and
# cross-scored). Arms: P0, GEOM8 (oracle), CTA4 (previous best, 55149 on the 55018 reader), CTAV2 / ENDV2 / DIRV2 (v2
# checkpoint V2=<train dir>), DINOWM (official DINO-WM PushT checkpoint through ti_wm/dinowm_scorer.py).
# EXTRA_DIRECT=NAME=path: also log a scripts/cta_direct_matched.py scorer.
# DRAW=64: policy samples per decision (K-scaling; candidates 0-7 are the K-bank, GEOM64 = geometry-best of all).
# Modes: smoke (unit tests + 2 roots x 2 decisions) | closed (array task t: roots FIRST + 25 t) | aggregate (CPU)
# | kscale (CPU, DRAW=64 analysis).
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
MODE=${1:?smoke/closed/aggregate}
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE="$TI/cta_v2cl_${MODE}_${JOB}"
RUN=$([[ "$MODE" == closed ]] && echo "$BASE/shard_$TASK" || echo "$BASE")
CODE="$RUN/code"
[[ ! -e "$CODE" ]] || { echo "Existing code snapshot: $CODE" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG=cta_v2cl
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
S="$CODE/scripts"
PARENT="$ROOT/cta_geometry_e2e_55018/train"
R4="$ROOT/cta_plus_train_55149/train"
DINOWM=/mnt/data/nhatnc129/jepa/dino_wm_official/outputs/pusht
# ARMS / LOG use ':' as separator in the environment (sbatch --export splits on commas)
ARMS=$(echo "${ARMS:-P0:GEOM8:CTA4:CTAV2:ENDV2:DIRV2:DINOWM}" | tr ':' ',')
LOG=$(echo "${LOG:-CTA4:CTAV2:ENDV2:DIRV2:DINOWM:FULLV2:CODEV2}" | tr ':' ',')
DW=$([[ "$LOG" == *DINOWM* ]] && echo "--dinowm $DINOWM" || true)
FIRST=${FIRST:-2200}
case "$MODE" in
  smoke)
    "$PY" -m unittest discover -s "$CODE/cta_tests" -p "test_dinowm_adapter.py"
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN/closed/shard_0" --parent "$PARENT" --r4 "$R4" \
      ${V2:+--v2 "$V2"} $DW ${EXTRA_DIRECT:+--extra-direct "$EXTRA_DIRECT"} --prep "$PREP" --smoke "$SMOKE" \
      --first "$FIRST" --count 2 --max-decisions 2 \
      --arms "$ARMS" --log-scorers "$LOG" ${DRAW:+--draw "$DRAW"}
    echo V2CL_SMOKE_OK
    ;;
  closed)
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r4 "$R4" --v2 "${V2:?V2}" \
      $DW ${EXTRA_DIRECT:+--extra-direct "$EXTRA_DIRECT"} --prep "$PREP" --smoke "$SMOKE" \
      --first "$((FIRST + TASK * 25))" --count 25 --arms "$ARMS" --log-scorers "$LOG" ${DRAW:+--draw "$DRAW"}
    ;;
  aggregate)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" --closed-run "$TI/cta_v2cl_closed_${2:?closed job}" \
      --pairs CTAV2-CTA4,CTAV2-DIRV2,CTAV2-ENDV2,CTAV2-DINOWM,CTAV2-GEOM8,DINOWM-P0
    ;;
  kscale)      # CPU: $2 = closed job id of a DRAW=64 run (scripts/cta_kscale_analysis.py; 55666 = its K=8 reference)
    "$PY" "$S/cta_kscale_analysis.py" --closed-run "$TI/cta_v2cl_closed_${2:?closed job}" \
      --reference-run "$TI/cta_v2cl_closed_${REF:-55666}" --out "$RUN"
    ;;
  *) echo "Unknown mode $MODE" >&2; exit 2 ;;
esac
