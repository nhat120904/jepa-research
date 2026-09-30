#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_plus
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_plus_%x_%A_%a.out
# docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md
# Usage: sbatch [resource overrides] slurm_cta_plus.sh <immutable release> <mode> [upstream job ids]
#   smoke                      unit tests + every new code path on tiny inputs (GPU)
#   diag      (array 0-3)      (1) on-policy diagnostic, roots 2100+25*task, 25 roots, 8 arms (GPU)
#   diag_agg  <diag id>        (1) aggregate (CPU)
#   collect   (array 0-23)     (2) new data, roots 31050+50*task, 50 roots (GPU)
#   encode    <collect id>     (2) compact features + frozen source codes (GPU)
#   train     <encode id>      (2) round-4 WM training + offline ladder (GPU)
#   r4closed  <train id> (array 0-3)  (2) closed loop of round-4 arms with logging (GPU)
#   r4_agg    <r4closed id> <diag id> (2) aggregate, merged with (1) when P0 trajectories match (CPU)
#   r5smoke                    Round 5 (16-candidate bank: policy + perturbed copies) code paths + demo, tiny (GPU)
#   r5closed  (array 0-7)      Round 5 closed loop, dev roots 2100+25*task (200 roots), round-4 nets (GPU)
#   r5_agg    <r5closed id>    Round 5 aggregate (CPU)
#   demo      <r5closed id>    PushT videos / stills of selected roots, P0 vs CTA4 (GPU)
#   r5pen     (array 0-1)      deviation-penalized CTA4 on the 16-bank, dev roots 2100-2149 (GPU)
#   r5pen_agg <r5pen id>       its aggregate (CPU)
#   r6smoke                    Round 6 (16-bank training) tiny train + closed loop + aggregate (GPU)
#   r6train                    Round 6 training: reader (stage A), CTA WM + direct (stage B), offline 16-bank table (GPU)
#   r6closed  <r6train id> (array 0-7)  Round 6 closed loop, dev roots 2100+25*task (GPU)
#   r6_agg    <r6closed id>    Round 6 aggregate (CPU)
#   r6demo    <r6train id> <r6closed id>  videos / stills, P0 vs CTA6 (GPU)
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?immutable release}
MODE=${2:?mode}
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE="$TI/cta_plus_${MODE}_${JOB}"
case "$MODE" in
  diag|r4closed|r5closed|r6closed|r5pen) RUN="$BASE/shard_$TASK"; CODE="$RUN/code"; HASHES="$RUN/CODE_SHA256SUMS" ;;
  collect)       RUN="$BASE"; CODE="$BASE/code_$TASK"; HASHES="$BASE/CODE_SHA256SUMS_$TASK" ;;
  *)             RUN="$BASE"; CODE="$RUN/code"; HASHES="$RUN/CODE_SHA256SUMS" ;;
esac
[[ ! -e "$CODE" ]] || exit 2
export CTA_TAG="plus_$MODE" WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$HASHES"
PARENT="$ROOT/cta_geometry_e2e_55018/train"
OLDF="$ROOT/cta_geometry_e2e_55018/features"
COLL="$ROOT/cta_collect_54489"
R3="$ROOT/cta_parallel_train_55077"
S="$CODE/scripts"
R4_ARMS=P0,PHYS8,GEOM8,CTA4,NLL4,DIRECT4,CTA4S
R4_LOG=CTA4,NLL4,DIRECT4,CTA4S,CTA3,DIRECT3,FULL
R4_PAIRS=CTA4-DIRECT4,CTA4-NLL4,CTA4-CTA4S,CTA4-CTA3,DIRECT4-DIRECT3,NLL4-NLL8,CTA4-GEOM8,DIRECT4-GEOM8
R4_TRAIN="$ROOT/cta_plus_train_55149/train"
R5_ARMS=P0,GEOM8,GEOM16,CTA4,CTA4@8,DIRECT4,NLL4,FRAME8
R5_LOG=CTA4,NLL4,DIRECT4,FRAME8,FULL
R5_PAIRS=CTA4-DIRECT4,CTA4-NLL4,CTA4-FRAME8,CTA4-CTA4@8,GEOM16-GEOM8,CTA4-GEOM16,DIRECT4-GEOM16
R4_FEAT="$ROOT/cta_plus_encode_55148/features"
PEN_ARMS=P0,CTA4~0.04,CTA4~0.08,CTA4~0.16
PEN_PAIRS=CTA4~0.08-CTA4~0.04,CTA4~0.16-CTA4~0.08
R6_ARMS=P0,GEOM16,CTA6,CTA6@8,DIRECT6,CODE6
R6_LOG=CTA6,DIRECT6,CODE6,CTA4,FULL
R6_PAIRS=CTA6-DIRECT6,CTA6-CTA6@8,CTA6-CODE6,CTA6-GEOM16,DIRECT6-GEOM16,CODE6-GEOM16
case "$MODE" in
  smoke)
    unit_tests
    W="$RUN/smoke"
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$W/diag/shard_0" --parent "$PARENT" --r3 "$R3" \
      --prep "$PREP" --smoke "$SMOKE" --first 2100 --count 2 --max-decisions 3
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$W/diag_agg" --closed-run "$W/diag"
    mkdir -p "$W/collect"
    "$PY" "$S/cta_collect_plus.py" --out "$W/collect" --prep "$PREP" --smoke "$SMOKE" --first 39994 --count 2 \
      --max-decisions 3
    "$PY" "$S/cta_encode_plus.py" --collect "$W/collect" --old-features "$OLDF" --parent "$PARENT" \
      --out "$W/features" --old-limit 64
    "$PY" "$S/cta_round4.py" --run "$W/train" --parent "$PARENT" --features "$W/features" --old-features "$OLDF" \
      --collection "$COLL" --r3 "$R3" --updates 20 --eval-every 10 --limit 64 --dev-limit 40
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$W/r4closed/shard_0" --parent "$PARENT" --r3 "$R3" \
      --r4 "$W/train" --prep "$PREP" --smoke "$SMOKE" --first 2100 --count 2 --max-decisions 3 \
      --arms "$R4_ARMS" --log-scorers "$R4_LOG"
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$W/r4_agg" --closed-run "$W/r4closed" "$W/diag" \
      --pairs "$R4_PAIRS"
    echo SMOKE_OK
    ;;
  diag)
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r3 "$R3" --prep "$PREP" \
      --smoke "$SMOKE" --first "$((2100 + TASK * 25))" --count 25
    ;;
  diag_agg)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" --closed-run "$TI/cta_plus_diag_${3:?diag id}" \
      --expect-roots 2100 2199
    ;;
  collect)
    "$PY" "$S/cta_collect_plus.py" --out "$RUN" --prep "$PREP" --smoke "$SMOKE" --first "$((31050 + TASK * 50))" \
      --count 50
    ;;
  encode)
    "$PY" "$S/cta_encode_plus.py" --collect "$TI/cta_plus_collect_${3:?collect id}" --old-features "$OLDF" \
      --parent "$PARENT" --out "$RUN/features"
    ;;
  train)
    "$PY" "$S/cta_round4.py" --run "$RUN/train" --parent "$PARENT" --features "$TI/cta_plus_encode_${3:?encode id}/features" \
      --old-features "$OLDF" --collection "$COLL" --r3 "$R3"
    ;;
  r4closed)
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r3 "$R3" \
      --r4 "$TI/cta_plus_train_${3:?train id}/train" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2100 + TASK * 25))" --count 25 --arms "$R4_ARMS" --log-scorers "$R4_LOG"
    ;;
  r4_agg)
    # optional 4th argument: a complete diagnostic run to merge (exact pairing only if P0 trajectories match)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" \
      --closed-run "$TI/cta_plus_r4closed_${3:?r4closed id}" ${4:+"$TI/cta_plus_diag_$4"} \
      --pairs "$R4_PAIRS" --expect-roots 2100 2199
    ;;
  r5smoke)
    unit_tests
    W="$RUN/smoke"
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$W/r5closed/shard_0" --parent "$PARENT" --r3 "$R3" \
      --r4 "$R4_TRAIN" --prep "$PREP" --smoke "$SMOKE" --first 2100 --count 2 --max-decisions 3 \
      --arms "$R5_ARMS" --log-scorers "$R5_LOG" --bank mixed
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$W/r5_agg" --closed-run "$W/r5closed" --pairs "$R5_PAIRS"
    "$PY" "$S/cta_demo.py" --run "$W/demo" --parent "$PARENT" --r3 "$R3" --r4 "$R4_TRAIN" --prep "$PREP" \
      --smoke "$SMOKE" --roots 2100 --max-decisions 4 --stills 0 2
    echo SMOKE_OK
    ;;
  r5closed)
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r3 "$R3" --r4 "$R4_TRAIN" \
      --prep "$PREP" --smoke "$SMOKE" --first "$((2100 + TASK * 25))" --count 25 --arms "$R5_ARMS" \
      --log-scorers "$R5_LOG" --bank mixed
    ;;
  r5_agg)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" --closed-run "$TI/cta_plus_r5closed_${3:?r5closed id}" \
      --pairs "$R5_PAIRS" --expect-roots 2100 2299
    ;;
  demo)
    "$PY" "$S/cta_demo.py" --run "$RUN" --parent "$PARENT" --r3 "$R3" --r4 "$R4_TRAIN" --prep "$PREP" \
      --smoke "$SMOKE" --from-run "$TI/cta_plus_r5closed_${3:?r5closed id}" --per-case 3
    ;;
  r5pen)
    unit_tests
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r4 "$R4_TRAIN" \
      --prep "$PREP" --smoke "$SMOKE" --first "$((2100 + TASK * 25))" --count 25 --arms "$PEN_ARMS" \
      --log-scorers CTA4 --bank mixed
    ;;
  r5pen_agg)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" --closed-run "$TI/cta_plus_r5pen_${3:?r5pen id}" \
      --pairs "$PEN_PAIRS" --expect-roots 2100 2149
    ;;
  r6smoke)
    unit_tests
    W="$RUN/smoke"
    "$PY" "$S/cta_round6.py" --run "$W/train" --parent "$PARENT" --features "$R4_FEAT" --r4 "$R4_TRAIN" \
      --updates-a 20 --updates-b 20 --eval-every 10 --limit 64
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$W/r6closed/shard_0" --parent "$PARENT" --r4 "$R4_TRAIN" \
      --r6 "$W/train" --prep "$PREP" --smoke "$SMOKE" --first 2100 --count 2 --max-decisions 3 \
      --arms "$R6_ARMS" --log-scorers "$R6_LOG" --bank mixed
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$W/r6_agg" --closed-run "$W/r6closed" --pairs "$R6_PAIRS"
    echo SMOKE_OK
    ;;
  r6train)
    "$PY" "$S/cta_round6.py" --run "$RUN/train" --parent "$PARENT" --features "$R4_FEAT" --r4 "$R4_TRAIN"
    ;;
  r6closed)
    "$PY" "$S/cta_diag_onpolicy.py" --mode closed --run "$RUN" --parent "$PARENT" --r4 "$R4_TRAIN" \
      --r6 "$TI/cta_plus_r6train_${3:?r6train id}/train" --prep "$PREP" --smoke "$SMOKE" \
      --first "$((2100 + TASK * 25))" --count 25 --arms "$R6_ARMS" --log-scorers "$R6_LOG" --bank mixed
    ;;
  r6_agg)
    "$PY" "$S/cta_diag_onpolicy.py" --mode aggregate --run "$RUN" --closed-run "$TI/cta_plus_r6closed_${3:?r6closed id}" \
      --pairs "$R6_PAIRS" --expect-roots 2100 2299
    ;;
  r6demo)
    "$PY" "$S/cta_demo.py" --run "$RUN" --parent "$PARENT" --r4 "$R4_TRAIN" --r6 "$TI/cta_plus_r6train_${3:?r6train id}/train" \
      --prep "$PREP" --smoke "$SMOKE" --from-run "$TI/cta_plus_r6closed_${4:?r6closed id}" --per-case 3 \
      --arms P0,CTA6 --log-scorers CTA6
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
