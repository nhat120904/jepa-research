#!/usr/bin/env bash
#SBATCH --job-name=hys_sw2
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_sw2_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
# With the P-space predictor added. Target: curvature well below the frozen 1.22
# while effective rank stays high (collapse floor previously ~3-4 of 256).
for PAIR in "10 10 5" "10 50 5" "50 50 5" "10 100 1" "50 200 1"; do
  set -- $PAIR; LC=$1; LP=$2; LV=$3
  echo "########## curve=$LC pred=$LP var=$LV ##########"
  .venv/bin/python $S/scripts/02_train_straightener.py \
    --cache $CACHE --tasks mw-push --gate none \
    --lambda-curve $LC --lambda-pred $LP --lambda-var $LV \
    --epochs 12 --seed 0 \
    --out $S/outputs/sw2_lc${LC}_lp${LP}_lv${LV}.pt 2>&1 | grep -E "^gate=|action_dim|init |ep00 |ep05 |ep11 "
done
echo "===== SWEEP2_DONE ===== $(date)"
