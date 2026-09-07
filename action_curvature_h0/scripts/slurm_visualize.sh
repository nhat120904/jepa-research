#!/usr/bin/env bash
#SBATCH --job-name=acm_viz
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_viz_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/action_curvature_h0"
PERD="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
PY="$STAGE0_ROOT/.venv/bin/python"
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/visualize_curvature.py" "$PROJECT/scripts/compose_curvature_viz.py"

# Two contrasting states from the held-out set, chosen by measured curvature
# per unit probe, NOT by anything downstream:
#   117 = lowest  (0.0020, object travels 0.018 mm -- effectively still)
#   093 = highest (0.9727, object travels 1.2 mm)  -> 478x apart
for S in 117 093; do
  echo "=== snapshot $S $(date -u +%FT%TZ) ==="
  "$PY" "$PROJECT/scripts/visualize_curvature.py" \
    --snapshot-index "$S" --manifest "$PERD/outputs/h0/manifest.json" \
    --population-dir "$PROJECT/outputs/cem_populations" \
    --action-source dataset --sigma 0.2 \
    --out-dir "$PROJECT/outputs/viz"
done

echo "=== compose (no GPU needed, run here for convenience) ==="
"$PY" "$PROJECT/scripts/compose_curvature_viz.py" \
  --viz "$PROJECT/outputs/viz" --low 117 --high 93 \
  --out-dir "$PROJECT/outputs/figures"
