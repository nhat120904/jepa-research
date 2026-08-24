#!/usr/bin/env bash
#SBATCH --job-name=hys_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_train_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
TASKS="mw-push mw-pick-place mw-reach"
EPOCHS=${EPOCHS:-40}
SEED=${SEED:-0}
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date) epochs=$EPOCHS seed=$SEED"

for GATE in none switch off; do
  echo "########## gate=$GATE ##########"
  .venv/bin/python $S/scripts/02_train_straightener.py \
    --cache $CACHE --tasks $TASKS --gate $GATE \
    --lambda-curve 10 --lambda-pred 10 --lambda-recon 200 --lambda-var 5 --epochs $EPOCHS --seed $SEED \
    --out $S/outputs/projector_${GATE}_seed${SEED}.pt
done
echo "===== HYS_TRAIN_DONE ===== $(date)"
