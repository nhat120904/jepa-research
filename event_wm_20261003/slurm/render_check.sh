#!/usr/bin/env bash
#SBATCH --job-name=ew_render
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Which MuJoCo rendering backend reproduces the dataset frames? (both venvs x egl/osmesa)
set -uo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/render_${SLURM_JOB_ID}
CACHE=$RUN_ROOT/cache/visual-puzzle-4x5-play-v0
for PY in torch ogb; do
  for GL in egl osmesa; do
    BIN=$TORCH_PY; [ $PY = ogb ] && BIN=$OGB_PY
    echo "== $PY $GL"
    MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL timeout 300 "$BIN" "$PROJECT/scripts/render_check.py" --cache "$CACHE" --out "$OUT/${PY}_${GL}" 2>&1 | grep -v Warn | tail -3
  done
done
