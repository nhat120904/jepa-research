#!/usr/bin/env bash
#SBATCH --job-name=tdj_wrk
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/tdj_wrk_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
TDJ="$REPO/diagnosis/external/td-jepa"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export MKL_NUM_THREADS=1 WANDB_MODE=disabled
cd "$TDJ"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "TDJEPA_COMMIT=$(git rev-parse HEAD)"

# num_workers is a dataloader parallelism knob, but it is also a value in the
# official recipe (config/train/lewm.yaml: num_workers 2).  The fidelity gate is
# the one place a recipe deviation is unacceptable, so it is only allowed if it
# is PROVEN not to change the computation.  Run the identical 200 steps at
# num_workers 2 and 8 and compare the logged losses step by step: identical
# losses prove the knob is pure I/O and the speedup may be taken.
OUT=/mnt/data/nhatnc129/jepa_runs/logs/tdj_wrk_${SLURM_JOB_ID:-manual}
for NW in 2 8; do
  echo "=== num_workers=$NW start $(date -u +%FT%TZ) ==="
  # Full output to a file, never through a pipe: a pipe both hides the traceback
  # and, under `set -o pipefail`, kills the job before it can be read.
  OMP_NUM_THREADS=1 "$STAGE0_ROOT/.venv/bin/python" train.py --config-name=ogb_train \
    data=ogb variant=td_jepa seed=3072 \
    ~data.dataset.rdcc_nbytes ~data.dataset.rdcc_w0 \
    num_workers=$NW \
    trainer.max_epochs=1 +trainer.limit_train_batches=200 \
    wandb.enabled=false planning_eval.enabled=false \
    output_model_name=workers/nw${NW}_200steps > "${OUT}_nw${NW}.log" 2>&1 || {
      echo "num_workers=$NW FAILED; last 30 lines:"; tail -30 "${OUT}_nw${NW}.log"; exit 1; }
  echo "=== num_workers=$NW end   $(date -u +%FT%TZ) ==="
  grep -E "step [0-9]+/|loss" "${OUT}_nw${NW}.log" | tail -10 || true
done
echo "=== step-by-step loss comparison ==="
"$STAGE0_ROOT/.venv/bin/python" - "${OUT}_nw2.log" "${OUT}_nw8.log" <<'PYEOF'
import re, sys
def losses(path):
    pat = re.compile(r"loss[\"'=:\s]+([0-9]*\.?[0-9]+)")
    return [float(m.group(1)) for m in pat.finditer(open(path, errors="ignore").read())]
a, b = losses(sys.argv[1]), losses(sys.argv[2])
print(f"nw2 losses parsed: {len(a)}   nw8 losses parsed: {len(b)}")
if not a or not b:
    print("NO LOSSES PARSED - comparison inconclusive, deviation refused")
    raise SystemExit(0)
n = min(len(a), len(b))
diff = max(abs(x - y) for x, y in zip(a[:n], b[:n]))
print(f"compared {n} values, max |difference| = {diff:.3e}")
print("IDENTICAL - num_workers is pure I/O" if diff == 0 else
      ("WITHIN 1e-6" if diff < 1e-6 else "DIVERGES - deviation refused"))
PYEOF
