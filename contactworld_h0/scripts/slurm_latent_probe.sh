#!/usr/bin/env bash
#SBATCH --job-name=cw_latent
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/latent_%j.out
set -euo pipefail
V=/mnt/data/nhatnc129/contactworld/cw_offline_venv
S=/home/nhatnc129/nhat.nc/jepa-research/contactworld_h0
C=/mnt/data/nhatnc129/contactworld/ContactWorld
cd $C
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"

for TASK in exploration_search insertion_usb; do
  echo "===== $TASK :: pointcloud only ====="
  $V/bin/python $S/scripts/02_latent_probe.py --task $TASK \
    --ckpt $C/logs/ckpts/$TASK/pointcloud/vc/only/100000.ckpt \
    --out $S/outputs/latent_${TASK}_pc.json || echo "FAILED pc $TASK"

  echo "===== $TASK :: pointcloud + TacFF ====="
  $V/bin/python $S/scripts/02_latent_probe.py --task $TASK \
    --ckpt $C/logs/ckpts/$TASK/pointcloud/vc/tactile_force_field_right/concat/100000.ckpt \
    --use-tactile \
    --out $S/outputs/latent_${TASK}_pc_ff.json || echo "FAILED pc_ff $TASK"
done
echo "===== LATENT_DONE ===== $(date)"
