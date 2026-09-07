#!/usr/bin/env bash
#SBATCH --job-name=spwm_bench2
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_bench2_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
sha256sum "$PROJECT/scripts/bench_render2.py" "$PROJECT/scripts/slurm_bench_render2.sh"
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no gpu (rendering is software)"
echo "--- EGL vendor probe ---"
ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || true
ls /usr/share/vulkan/icd.d/ 2>/dev/null || true
"$PY" "$PROJECT/scripts/bench_render2.py" --n 20
echo "DONE $(date -u +%FT%TZ)"
