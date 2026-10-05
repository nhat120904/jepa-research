#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_reader_closed
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_reader_closed_%x_%A_%a.out
# CPU defaults. closed requires explicit GPU resource overrides, for example:
#   sbatch --partition=mig --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
#     --cpus-per-task=4 --mem=64G --time=00:45:00 --array=0-1%1 \
#     --export=ALL,CTA_SOURCE_ROOT=/immutable/release/code \
#     slurm_cta_reader_closed.sh closed <completed reader-training job id>
#   sbatch --dependency=afterok:<closed array id> --export=ALL,CTA_SOURCE_ROOT=/immutable/release/code \
#     slurm_cta_reader_closed.sh aggregate <closed array id>
# Submit only after scheduler/artifact verification and the account usage check.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch' >&2; exit 1; }
PROJECT=${CTA_SOURCE_ROOT:?Set CTA_SOURCE_ROOT to the immutable release/code directory}
MODE=${1:?mode: closed or aggregate}
UPSTREAM=${2:?completed reader-training ID for closed, closed-array ID for aggregate}
[[ "$UPSTREAM" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric upstream job ID' >&2; exit 2; }
TASK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
BASE=${BASE_CHECKPOINT:-$TI/cta_replan15_train_56504/train/cta_v2.pt}
NATIVE_TRAIN_JOB=${NATIVE_TRAIN_JOB:-56914}
[[ "$NATIVE_TRAIN_JOB" =~ ^[0-9]+$ ]] || { echo 'NATIVE_TRAIN_JOB must be numeric' >&2; exit 2; }
NATIVE="$TI/cta_hit_train_${NATIVE_TRAIN_JOB}/train"
FIRST=${FIRST:-2220}
COUNT=${COUNT:-10}
ARMS=${ARMS:-P0,GEOM8,CTA_BASE,CTA_WMCTRL,END_HIT,DIR_HIT,CTA_RCTRL,CTA_RHIT}
ARMS=${ARMS//:/,}
PAIRS=${PAIRS:-CTA_RHIT-CTA_RCTRL,CTA_RHIT-CTA_BASE,CTA_RHIT-CTA_WMCTRL,CTA_RHIT-END_HIT,CTA_RHIT-DIR_HIT,CTA_RHIT-P0,CTA_RCTRL-CTA_BASE}
PAIRS=${PAIRS//:/,}
case "$MODE" in
  closed)
    [[ "$TASK" =~ ^[01]$ ]] || { echo 'Reader pilot expects array tasks 0 and 1' >&2; exit 2; }
    [[ "$FIRST" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || { echo 'FIRST and COUNT must be numeric' >&2; exit 2; }
    [[ "$COUNT" -ge 2 ]] || { echo 'Each shard needs at least two roots for qualification' >&2; exit 2; }
    [[ -n "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}" ]] || {
      echo 'closed requires explicit GPU sbatch resources and a 45-minute limit' >&2; exit 2;
    }
    RUN="$TI/cta_reader_closed_${JOB}/shard_${TASK}"
    ;;
  aggregate) RUN="$TI/cta_reader_aggregate_${JOB}" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || { echo "Refusing to overwrite $RUN" >&2; exit 2; }
export WANDB_MODE=disabled CTA_TAG=reader15
source "$PROJECT/scripts/cta_env.sh"
mkdir -p "$RUN"
snapshot "$RUN/CODE_SHA256SUMS"
S="$CODE/scripts"
printf 'source=%s mode=%s job=%s task=%s upstream=%s native_train=%s\n' \
  "$PROJECT" "$MODE" "$JOB" "$TASK" "$UPSTREAM" "$NATIVE_TRAIN_JOB" > "$RUN/launch.txt"
case "$MODE" in
  closed)
    TRAIN="$TI/cta_reader_train_${UPSTREAM}/train"
    # Metadata only: refuse unfinished calibration or two unchanged fallbacks.
    # Model loading/physics happen below, within this explicit GPU allocation.
    "$PY" - "$TRAIN" "$UPSTREAM" "$BASE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

train, job, base = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
config = json.loads((train / "config.json").read_text())["reader_calibration"]
selection = json.loads((train / "selection.json").read_text())
finished = [json.loads(line) for line in (train / "metrics.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line).get("status") == "DONE"]
if len(finished) != 1 or str(config["job"]) != job:
    raise SystemExit("Reader calibration lacks one matching DONE record")
done = finished[0]
if done["updates"] != config["steps"] or selection["step"] != config["steps"]:
    raise SystemExit("Reader calibration did not finish the configured update budget")
steps = selection["selected_steps"]
if steps != done["selected_steps"] or steps != config["selected_steps"]:
    raise SystemExit("Reader selected-state provenance differs across metadata files")
if set(steps) != {"reader_control", "reader_hit"} or any(type(v) is not int or v < 0 for v in steps.values()):
    raise SystemExit("Invalid paired reader selected-step mapping")
if not config.get("frozen_check") or not all(config["frozen_check"].values()):
    raise SystemExit("Reader trainer did not verify frozen encoder and world model")
if done.get("frozen_check") != config["frozen_check"]:
    raise SystemExit("Reader frozen-state provenance differs across metadata files")
digest = hashlib.sha256()
with base.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != config["base_sha256"]:
    raise SystemExit("Reader calibration BASE checkpoint changed")
for variant in ("control", "hit"):
    if not (train / variant / "cta_v2.pt").is_file():
        raise SystemExit(f"Missing minimal {variant} reader checkpoint")
if all(step == 0 for step in steps.values()):
    raise SystemExit("Both readers selected BASE step0; skip duplicate GPU evaluation")
print(json.dumps({"reader_training_job": job, "selected_steps": steps,
                  "frozen_encoder_wm_verified": True}), flush=True)
PY
    "$PY" "$S/cta_hit_closed.py" --run "$RUN/results" \
      --checkpoint "BASE=$BASE" \
      --checkpoint "CTRL=$NATIVE/control/cta_v2.pt" \
      --checkpoint "HIT=$NATIVE/hit/cta_v2.pt" \
      --checkpoint "RCTRL=$TRAIN/control/cta_v2.pt" \
      --checkpoint "RHIT=$TRAIN/hit/cta_v2.pt" \
      --arm-spec 'CTA_BASE=BASE:cta' \
      --arm-spec 'CTA_WMCTRL=CTRL:cta' \
      --arm-spec 'END_HIT=HIT:endpoint' \
      --arm-spec 'DIR_HIT=HIT:direct' \
      --arm-spec 'CTA_RCTRL=RCTRL:cta' \
      --arm-spec 'CTA_RHIT=RHIT:cta' \
      --arms "$ARMS" --first "$((FIRST + TASK * COUNT))" --count "$COUNT" --n-exec 15
    ;;
  aggregate)
    CLOSED="$TI/cta_reader_closed_${UPSTREAM}"
    shopt -s nullglob
    INPUTS=("$CLOSED"/shard_*/results)
    [[ ${#INPUTS[@]} -gt 0 ]] || { echo "No reader closed-loop outputs in $CLOSED" >&2; exit 2; }
    "$PY" "$S/cta_hit_aggregate.py" --closed-run "${INPUTS[@]}" --out "$RUN/aggregate" \
      --pairs "$PAIRS" --expect-roots "${EXPECTED_FIRST:-2220}" "${EXPECTED_LAST:-2239}"
    ;;
esac
echo "DONE $MODE $(date -u +%FT%TZ)"
