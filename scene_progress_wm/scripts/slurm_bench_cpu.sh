#!/usr/bin/env bash
#SBATCH --job-name=spwm_benchcpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_benchcpu_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
echo "=== no GPU requested; nvidia-smi should be unavailable or show nothing ==="
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no nvidia-smi"
sha256sum "$PROJECT/scripts/bench_render.py"
"$PY" "$PROJECT/scripts/bench_render.py" --n 50 --render-size 64
echo "DONE $(date -u +%FT%TZ)"
