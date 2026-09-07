#!/usr/bin/env bash
#SBATCH --job-name=spwm_bench
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_bench_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
sha256sum "$PROJECT/scripts/bench_render.py" "$PROJECT/scripts/slurm_bench_render.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/bench_render.py" --n 200 --render-size 64
echo "DONE $(date -u +%FT%TZ)"
