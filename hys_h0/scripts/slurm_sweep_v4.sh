#!/usr/bin/env bash
#SBATCH --job-name=hys_sw4
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_sw4_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
# Does straightening keep the object? Effective rank was the wrong guard.
for LC in 0 2 10 50; do
  echo "########## curve=$LC ##########"
  .venv/bin/python $S/scripts/02_train_straightener.py \
    --cache $CACHE --tasks mw-push --gate none \
    --lambda-curve $LC --lambda-pred 10 --lambda-recon 200 --lambda-var 5 \
    --epochs 12 --seed 0 \
    --out $S/outputs/sw4_lc${LC}.pt 2>&1 | grep -E "^gate=|init |ep11 |OBJECT"
done
echo "===== SWEEP4_DONE ===== $(date)"
