# Sourced by the CTA Slurm scripts after RUN and CODE are set (docs/CTA_E2E_PROTOCOL.md).
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
PREP="$ROOT/prepare_53776"
SMOKE="$ROOT/gate_smoke_53803"
export PYTHONPATH="$CODE:$PREP/deps:$PREP/upstream"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export WANDB_PROJECT="${WANDB_PROJECT:-cta-pusht}" WANDB_DIR="$RUN" WANDB_SILENT=true CTA_TAG="${CTA_TAG:-r0}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if { [[ -n "${WANDB_API_KEY:-}" ]] || grep -qs api.wandb.ai "$HOME/.netrc"; } \
     && timeout 5 curl -s -o /dev/null https://api.wandb.ai; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi
echo "wandb: mode $WANDB_MODE, project $WANDB_PROJECT, group $CTA_TAG, dir $WANDB_DIR"

snapshot() {  # copy the code once per task and record its hashes; $1 = hash file
  mkdir -p "$CODE"
  cp -r "$PROJECT/ti_wm" "$PROJECT/tests" "$PROJECT/cta_tests" "$PROJECT/scripts" "$PROJECT/docs" "$CODE/"
  find "$CODE" -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$1"
}

unit_tests() {
  "$PY" -m unittest discover -s "$CODE/tests"
  "$PY" -m unittest discover -s "$CODE/cta_tests"
}
