#!/usr/bin/env bash
#SBATCH --job-name=acm_eqv
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_eqv_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
# Re-collect ONE LeWM snapshot through the new unified encode path.  The stored
# LeWM shards were produced by mc.encode_images_dtype; if the two paths do not
# agree, comparing LeWM against DINO-WM would compare encoders AND encoding
# code, so this is checked rather than assumed.
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/gate0_collect.py" \
  --snapshot-index 64 --manifest "$PERD/outputs/h0/manifest.json" \
  --populations-dir "$PROJECT/outputs/cem_populations" \
  --population-index 1 --n-candidates 12 \
  --out-dir "$PROJECT/outputs/gate0_lewm_recheck"
"$STAGE0_ROOT/.venv/bin/python" - <<'PY'
import numpy as np
a=np.load("/home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/outputs/gate0/snapshot_064/gate0.npz")["z"]
b=np.load("/home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/outputs/gate0_lewm_recheck/snapshot_064/gate0.npz")["z"]
print("stored", a.shape, "recheck", b.shape)
if a.shape==b.shape:
    d=np.abs(a-b).max(); s=np.abs(a).max()
    print(f"max |diff| = {d:.3e}   scale {s:.3e}   relative {d/max(s,1e-12):.3e}")
    print("EQUIVALENT" if d/max(s,1e-12) < 1e-4 else "DIFFERENT -> LeWM must be re-collected")
else:
    print("SHAPE MISMATCH -> LeWM must be re-collected")
PY
