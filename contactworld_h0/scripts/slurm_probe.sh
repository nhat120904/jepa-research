#!/usr/bin/env bash
#SBATCH --job-name=cw_probe
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/probe_%j.out
set -euo pipefail
V=/mnt/data/nhatnc129/contactworld/cw_offline_venv
S=/home/nhatnc129/nhat.nc/jepa-research/contactworld_h0
cd /mnt/data/nhatnc129/contactworld/ContactWorld
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"

for TASK in exploration_search insertion_usb; do
  if [ ! -d "data/demo_data/$TASK" ]; then echo "SKIP $TASK (not extracted)"; continue; fi
  echo "===== $TASK ====="
  $V/bin/python $S/scripts/01_observability_probe.py \
    --data-root data/demo_data/$TASK --task $TASK --target-key plug_pos \
    --seeds 0 1 2 --epochs 40 \
    --out $S/outputs/probe_${TASK}_plugpos.json
done
echo "===== PROBE_DONE ===== $(date)"
