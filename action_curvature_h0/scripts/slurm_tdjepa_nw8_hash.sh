#!/usr/bin/env bash
#SBATCH --job-name=tdj_nw8
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_nw8_%j.out
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

# The decisive test that was never run: does num_workers=8 feed the model the
# SAME batches as the recipe?  The earlier rejection used a weight criterion
# that has since been shown to have no discriminating power on this stack.
REF=/mnt/data/nhatnc129/jepa_runs/tdj_matrix_46938/A1_hashes.json
OUTJ=/mnt/data/nhatnc129/jepa_runs/tdj_nw8_${SLURM_JOB_ID:-manual}_hashes.json
BATCH_HASH_OUT="$OUTJ" "$PY" "$PROJECT/scripts/tdjepa_batch_hash_probe.py" \
  --config-name=ogb_train data=ogb variant=td_jepa seed=3072 \
  ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
  num_workers=8 \
  trainer.max_epochs=1 +trainer.limit_train_batches=200 \
  wandb.enabled=false planning_eval.enabled=false \
  output_model_name=matrix/NW8 > /mnt/data/nhatnc129/jepa_runs/tdj_nw8_${SLURM_JOB_ID}.log 2>&1 || {
    echo "FAILED; last 25:"; tail -25 /mnt/data/nhatnc129/jepa_runs/tdj_nw8_${SLURM_JOB_ID}.log; exit 1; }
grep -oE "step [0-9]+/200 \([0-9.]+ it/s\)" /mnt/data/nhatnc129/jepa_runs/tdj_nw8_${SLURM_JOB_ID}.log | tail -1

"$PY" - "$REF" "$OUTJ" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["hashes"]
b = json.load(open(sys.argv[2]))["hashes"]
n = min(len(a), len(b))
first = next((i for i in range(n) if a[i] != b[i]), None)
print(f"A1 hashes: {len(a)}   nw8 hashes: {len(b)}   compared: {n}")
print(f"identical: {a[:n] == b[:n]}")
print(f"first divergent batch: {first}")
print("VERDICT:", "SAME BATCH STREAM" if first is None and n > 0
      else f"DIFFERENT BATCH STREAM at batch {first}")
PYEOF
