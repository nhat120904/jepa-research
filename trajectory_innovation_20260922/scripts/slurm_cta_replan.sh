#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_replan
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_replan_%x_%A_%a.out
# docs/CTA_REPLAN_INTERVAL_PROTOCOL.md: CTA v2 pipeline at N_EXEC executed actions per decision (default 15).
# Usage: sbatch [resource overrides] slurm_cta_replan.sh <mode> [upstream job ids]
#   smoke                       unit tests + every stage on tiny inputs (GPU; code path only, not a result)
#   collect      (array 0-23)   train banks, roots 34000+50*task (GPU)
#   collect_dev  (array 0-1)    selection banks, roots 2000+50*task (GPU)
#   encode   <collect> <dev>    full-token features of both collections, original PCA (GPU)
#   train    <encode>           v2 recipe on these banks only (GPU)
#   closed   <train> (array 0-7) closed loop, roots 2200+25*task (GPU)
#   aggregate <closed>          cta_diag_onpolicy aggregate + replan read-out against 55666 (CPU)
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
MODE=${1:?mode}
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
N_EXEC=${N_EXEC:-15}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE="$TI/cta_replan${N_EXEC}_${MODE}_${JOB}"
case "$MODE" in
  collect|collect_dev) RUN="$BASE"; CODE="$BASE/code_$TASK"; HASHES="$BASE/CODE_SHA256SUMS_$TASK" ;;
  closed|headroom)     RUN="$BASE/shard_$TASK"; CODE="$RUN/code"; HASHES="$RUN/CODE_SHA256SUMS" ;;
  *)                   RUN="$BASE"; CODE="$RUN/code"; HASHES="$RUN/CODE_SHA256SUMS" ;;
esac
[[ ! -e "$CODE" ]] || { echo "Existing code snapshot: $CODE" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG="replan$N_EXEC"
source "$PROJECT/scripts/cta_env.sh"
mkdir -p "$RUN"
snapshot "$HASHES"
S="$CODE/scripts"
PCA="$ROOT/cta_feat_54490/pca.pt"
PARENT="$ROOT/cta_geometry_e2e_55018/train"
DINOWM=/mnt/data/nhatnc129/jepa/dino_wm_official/outputs/pusht
ARMS=P0,GEOM8,CTAV2,ENDV2,DIRV2,DINOWM
LOG=CTAV2,ENDV2,DIRV2,DINOWM,FULLV2,CODEV2
PAIRS=CTAV2-DIRV2,CTAV2-ENDV2,CTAV2-DINOWM,CTAV2-GEOM8,DINOWM-P0
collect() {   # $1 out dir, $2 first root, $3 count, [$4 max decisions]
  mkdir -p "$1"
  "$PY" "$S/cta_collect_plus.py" --out "$1" --prep "$PREP" --smoke "$SMOKE" --first "$2" --count "$3" \
    --n-exec "$N_EXEC" ${4:+--max-decisions "$4"}
}
encode() {    # $1 collection dir, $2 out dir, [$3 max shards]
  "$PY" "$S/cta_encode_r4_full.py" --collect "$1" --pca "$PCA" --out "$2" ${3:+--max-shards "$3"}
}
case "$MODE" in
  smoke)
    unit_tests
    collect "$RUN/collect" 39980 2 6
    collect "$RUN/collect_dev" 39982 2 6
    encode "$RUN/collect" "$RUN/feat"
    encode "$RUN/collect_dev" "$RUN/feat_dev"
    "$PY" "$S/cta_train_v2.py" --run "$RUN/train" --r4 "$RUN/feat" --with-perturbed --only-r4 \
      --select-r4 "$RUN/feat_dev" --steps1 20 --steps2 20 --eval-every 10 --select-limit 8 --final-limit 8
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN/closed/shard_0" --parent "$PARENT" --v2 "$RUN/train" \
      --dinowm "$DINOWM" --prep "$PREP" --smoke "$SMOKE" --first 2200 --count 2 --max-decisions 2 \
      --arms "$ARMS" --log-scorers "$LOG" --n-exec "$N_EXEC"
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN/aggregate" --closed-run "$RUN/closed" --pairs "$PAIRS"
    "$PY" "$S/cta_replan_analysis.py" --run15 "$RUN/closed" --out "$RUN/replan" --first 2200 --last 2201
    echo REPLAN_SMOKE_OK
    ;;
  collect)
    [[ "$TASK" -ge 0 && "$TASK" -le 23 ]] || exit 2
    collect "$BASE" "$((34000 + TASK * 50))" 50
    ;;
  collect_dev)
    [[ "$TASK" -ge 0 && "$TASK" -le 1 ]] || exit 2
    collect "$BASE" "$((2000 + TASK * 50))" 50
    ;;
  encode)
    encode "$TI/cta_replan${N_EXEC}_collect_${2:?collect id}" "$RUN/feat"
    encode "$TI/cta_replan${N_EXEC}_collect_dev_${3:?collect_dev id}" "$RUN/feat_dev"
    ;;
  train)
    F="$TI/cta_replan${N_EXEC}_encode_${2:?encode id}"
    "$PY" "$S/cta_train_v2.py" --run "$RUN/train" --r4 "$F/feat" --with-perturbed --only-r4 --select-r4 "$F/feat_dev" \
      --seed "${SEED:-0}"
    ;;
  closed)
    [[ "$TASK" -ge 0 && "$TASK" -le 7 ]] || exit 2
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" \
      --v2 "$TI/cta_replan${N_EXEC}_train_${2:?train id}/train" --dinowm "$DINOWM" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2200 + TASK * 25))" --count 25 --arms "$ARMS" --log-scorers "$LOG" --n-exec "$N_EXEC"
    ;;
  headroom)    # arms that need no trained scorer: P0 and the geometry oracle, roots 2200+100*task (array 0-1)
    [[ "$TASK" -ge 0 && "$TASK" -le 1 ]] || exit 2
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2200 + TASK * 100))" --count 100 --arms P0,GEOM8 --log-scorers "" --n-exec "$N_EXEC"
    ;;
  aggregate)
    C="$TI/cta_replan${N_EXEC}_closed_${2:?closed id}"
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN/aggregate" --closed-run "$C" --pairs "$PAIRS" \
      --expect-roots 2200 2399
    "$PY" "$S/cta_replan_analysis.py" --run8 "$TI/cta_v2cl_closed_55666" --run15 "$C" --out "$RUN/replan"
    ;;
  *) echo "Unknown mode $MODE" >&2; exit 2 ;;
esac
echo "DONE $MODE $(date -u +%FT%TZ)"
