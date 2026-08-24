#!/usr/bin/env bash
#SBATCH --job-name=hys_sw3
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_sw3_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
# With the invertibility (reconstruction) constraint. Question: does any setting reach
# curvature well below the frozen 1.221 while effective rank stays high?
for PAIR in "0 200" "10 200" "10 50" "50 200" "2 200"; do
  set -- $PAIR; LC=$1; LR=$2
  echo "########## curve=$LC recon=$LR ##########"
  .venv/bin/python $S/scripts/02_train_straightener.py \
    --cache $CACHE --tasks mw-push --gate none \
    --lambda-curve $LC --lambda-pred 10 --lambda-recon $LR --lambda-var 5 \
    --epochs 12 --seed 0 \
    --out $S/outputs/sw3_lc${LC}_lr${LR}.pt 2>&1 | grep -E "^gate=|lambda_curve|init |ep00 |ep05 |ep11 "
done
echo "===== SWEEP3_DONE ===== $(date)"
