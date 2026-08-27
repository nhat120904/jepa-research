#!/usr/bin/env bash
#SBATCH --job-name=tdj_bnd
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_bnd_%j.out
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

# Job 46962 proved num_workers=8 reproduces the batch stream for the first 200
# batches of epoch 0 -- which never crosses a reshuffle.  Ten epochs do.  This
# runs TWO epochs of 50 batches at num_workers 2 and 8 and compares all 100
# hashes, so the epoch boundary itself is inside the compared range.
REF=/mnt/data/nhatnc129/jepa_runs/tdj_bnd_${SLURM_JOB_ID}_nw2.json
CAND=/mnt/data/nhatnc129/jepa_runs/tdj_bnd_${SLURM_JOB_ID}_nw8.json
export BATCH_HASH_LIMIT=100

for NW in 2 8; do
  case $NW in 2) OUTJ=$REF;; 8) OUTJ=$CAND;; esac
  echo "=== num_workers=$NW $(date -u +%FT%TZ) ==="
  BATCH_HASH_OUT="$OUTJ" "$PY" "$PROJECT/scripts/tdjepa_batch_hash_probe.py" \
    --config-name=ogb_train data=ogb variant=td_jepa seed=3072 \
    ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
    num_workers=$NW \
    trainer.max_epochs=2 +trainer.limit_train_batches=50 \
    wandb.enabled=false planning_eval.enabled=false \
    output_model_name=boundary/nw${NW} \
    > /mnt/data/nhatnc129/jepa_runs/tdj_bnd_${SLURM_JOB_ID}_nw${NW}.log 2>&1 || {
      echo "nw=$NW FAILED; last 25:"
      tail -25 /mnt/data/nhatnc129/jepa_runs/tdj_bnd_${SLURM_JOB_ID}_nw${NW}.log; exit 1; }
done

"$PY" - "$REF" "$CAND" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["hashes"]
b = json.load(open(sys.argv[2]))["hashes"]
n = min(len(a), len(b))
first = next((i for i in range(n) if a[i] != b[i]), None)
print(f"nw2: {len(a)} hashes   nw8: {len(b)} hashes   compared: {n}")
print(f"(50 batches per epoch, so index 50 is the first batch after a reshuffle)")
print(f"identical: {a[:n] == b[:n]}   first divergent: {first}")
print(f"epoch-0 block identical: {a[:50] == b[:50]}")
print(f"epoch-1 block identical: {a[50:n] == b[50:n]}")
print(f"epochs actually differ from each other (reshuffle happened): {a[:50] != a[50:100]}")
print("VERDICT:", "SAME BATCH STREAM ACROSS THE EPOCH BOUNDARY" if first is None and n >= 100
      else f"DIVERGES at batch {first}")
PYEOF
