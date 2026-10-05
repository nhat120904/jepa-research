#!/usr/bin/env bash
#SBATCH --job-name=ew_ucpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Small CPU diagnostic: SCRIPT=u_pos_jitter.py ARGS="--run ... --cache ..." sbatch slurm/u_cpu.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
"$TORCH_PY" "$PROJECT/scripts/${SCRIPT:?SCRIPT}" ${ARGS:-}
