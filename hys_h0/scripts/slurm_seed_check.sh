#!/usr/bin/env bash
#SBATCH --job-name=hys_seed
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_seed_%A_%a.out
# none vs switch differed hugely in rho_final (-0.011 vs +0.071) despite nearly
# identical curvature (0.484 vs 0.527) and object decodability (6.79 vs 6.73 cm).
# That is exactly the shape of a training-seed artefact, so replicate before believing.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
GATES=(none switch none switch); SEEDS=(1 1 2 2)
I=${SLURM_ARRAY_TASK_ID}; GATE=${GATES[$I]}; SEED=${SEEDS[$I]}
echo "HOST=$(hostname) gate=$GATE seed=$SEED $(date)"
.venv/bin/python $S/scripts/02_train_straightener.py \
  --cache $CACHE --tasks mw-push mw-pick-place mw-reach --gate $GATE \
  --lambda-curve 10 --lambda-pred 10 --lambda-recon 200 --lambda-var 5 \
  --epochs 40 --seed $SEED \
  --out $S/outputs/projector_${GATE}_seed${SEED}.pt 2>&1 | grep -E "^gate=|init |ep39 |OBJECT"
echo "SEED_CELL_DONE gate=$GATE seed=$SEED $(date)"
