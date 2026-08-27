#!/usr/bin/env bash
#SBATCH --job-name=tdj_mtx
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_mtx_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
TDJ="$REPO/diagnosis/external/td-jepa"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_MODE=disabled
export TDJEPA_TRAIN="$TDJ/train.py" BATCH_HASH_LIMIT=200
cd "$TDJ"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "TDJEPA_COMMIT=$(git rev-parse HEAD)"
sha256sum "$PROJECT/scripts/tdjepa_batch_hash_probe.py"

WORK=/mnt/data/nhatnc129/jepa_runs/tdj_matrix_${SLURM_JOB_ID:-manual}
mkdir -p "$WORK"

# A1 is the official recipe; A2 repeats it to measure the reproducibility floor,
# without which "the weights differ" is not a valid criterion at all.  B and C
# each change ONE loader knob that is HYPOTHESISED to preserve sample order --
# that hypothesis is exactly what this measures, it is not assumed.
run () {  # name, extra hydra overrides...
  local NAME="$1"; shift
  echo "=== $NAME start $(date -u +%FT%TZ) ==="
  BATCH_HASH_OUT="$WORK/${NAME}_hashes.json" \
  "$PY" "$PROJECT/scripts/tdjepa_batch_hash_probe.py" --config-name=ogb_train \
    data=ogb variant=td_jepa seed=3072 \
    ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
    trainer.max_epochs=1 +trainer.limit_train_batches=200 \
    wandb.enabled=false planning_eval.enabled=false \
    "$@" output_model_name=matrix/${NAME} > "$WORK/${NAME}.log" 2>&1 || {
      echo "$NAME FAILED; last 25 lines:"; tail -25 "$WORK/${NAME}.log"; exit 1; }
  echo "=== $NAME end   $(date -u +%FT%TZ) ==="
  grep -oE "step [0-9]+/200 \([0-9.]+ it/s\)" "$WORK/${NAME}.log" | tail -1 || true
}

run A1
run A2
run B  loader.prefetch_factor=8
run C  loader.pin_memory=false

echo "=== comparison ==="
"$PY" "$PROJECT/scripts/compare_loader_matrix.py" \
  --work "$WORK" --checkpoints "$STABLEWM_HOME/checkpoints/matrix" \
  --out "$PROJECT/outputs/loader_matrix.json"
