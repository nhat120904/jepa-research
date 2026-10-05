#!/usr/bin/env bash
#SBATCH --job-name=ew_hcap
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Capacity check: can the heuristic's network family regress exact Lights Out distances? (diagnostic)
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/hcap_${SLURM_JOB_ID}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/h_capacity.py" --rows ${ROWS:-4} --cols ${COLS:-5} --out "$OUT"
