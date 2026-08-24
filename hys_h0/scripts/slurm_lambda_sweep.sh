#!/usr/bin/env bash
#SBATCH --job-name=hys_sweep
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_sweep_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
# lambda_curve : lambda_var pairs. Baseline run used 1:25 and the VICReg term won.
for PAIR in "1 25" "10 5" "50 5" "50 1" "200 1"; do
  set -- $PAIR; LC=$1; LV=$2
  echo "########## lambda_curve=$LC lambda_var=$LV ##########"
  .venv/bin/python $S/scripts/02_train_straightener.py \
    --cache $CACHE --tasks mw-push --gate none \
    --lambda-curve $LC --lambda-var $LV --epochs 12 --seed 0 \
    --out $S/outputs/sweep_lc${LC}_lv${LV}.pt 2>&1 | grep -E "^gate=|init |ep00 |ep05 |ep11 "
done
echo "===== SWEEP_DONE ===== $(date)"
