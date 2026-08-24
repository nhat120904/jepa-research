#!/usr/bin/env bash
#SBATCH --job-name=hys_v2
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --array=0-11%3
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_v2_%A_%a.out
#
# Series v2: adds the boundary head (beta*CE(q(P_t,P_t1), s_t)) from the original
# HyS-JEPA proposal, and the `random` control that drops a MATCHED fraction of
# transitions at random. If `random` matches `switch`, the contact semantics are
# irrelevant and the effect is just dropping the high-curvature tail.
# Not comparable to the v1 projectors (different objective) -- this is its own series.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
CACHE=data/precomputed_latents/metaworld__dino_wm_metaworld.h5
GATES=(none switch random off none switch random off none switch random off)
SEEDS=(0 0 0 0 1 1 1 1 2 2 2 2)
I=${SLURM_ARRAY_TASK_ID}; GATE=${GATES[$I]}; SEED=${SEEDS[$I]}
echo "HOST=$(hostname) gate=$GATE seed=$SEED $(date)"
.venv/bin/python $S/scripts/02_train_straightener.py \
  --cache $CACHE --tasks mw-push mw-pick-place mw-reach --gate $GATE \
  --lambda-curve 10 --lambda-pred 10 --lambda-recon 200 --lambda-var 5 --lambda-head 1 \
  --epochs 40 --seed $SEED \
  --out $S/outputs/v2_${GATE}_seed${SEED}.pt 2>&1 | grep -E "^gate=|switch_rate|init |ep39 |BOUNDARY|OBJECT"
echo "V2_CELL_DONE gate=$GATE seed=$SEED $(date)"
