#!/usr/bin/env bash
#SBATCH --job-name=ti_s1_check
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/s1_check_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
OUT=/mnt/data/nhatnc129/jepa/trajectory_innovation/s1_check_${SLURM_JOB_ID}
mkdir -p $OUT && cp /home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922/scripts/s1_signal_check.py $OUT/
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python $OUT/s1_signal_check.py /mnt/data/nhatnc129/jepa/trajectory_innovation/s1_collect_54418 $OUT
