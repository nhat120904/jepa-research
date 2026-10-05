#!/usr/bin/env bash
#SBATCH --job-name=ew_timing
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -uo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
CODE=$RUN_ROOT/code3_56938_visual-puzzle-4x5-play-v0/N64/code.pt
for GL in egl osmesa; do MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL "$TORCH_PY" "$PROJECT/scripts/step_timing.py" "$CODE" 2>&1 | grep '{'; done
