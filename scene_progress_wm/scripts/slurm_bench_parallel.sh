#!/usr/bin/env bash
#SBATCH --job-name=spwm_bpar
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_bpar_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
sha256sum "$PROJECT/scripts/bench_parallel.py" "$PROJECT/scripts/slurm_bench_parallel.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc
"$PY" "$PROJECT/scripts/bench_parallel.py" --frames 40 --size 64 --workers 1 4 8 16 32 --gl egl osmesa
echo "DONE $(date -u +%FT%TZ)"
