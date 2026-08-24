#!/usr/bin/env bash
#SBATCH --job-name=perd_submit_rest
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_submit_rest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
cd "$REPO"

COLLECT_B=$(sbatch --parsable "$PROJECT/scripts/slurm_01_collect_second.sh")
TRAIN_SMOKE=$(sbatch --parsable --dependency="afterok:$COLLECT_B" "$PROJECT/scripts/slurm_02_train_smoke.sh")
TRAIN=$(sbatch --parsable --dependency="afterok:$TRAIN_SMOKE" "$PROJECT/scripts/slurm_02_train.sh")
EVAL_SMOKE=$(sbatch --parsable --dependency="afterok:$TRAIN" "$PROJECT/scripts/slurm_03_eval_smoke.sh")
EVAL=$(sbatch --parsable --dependency="afterok:$EVAL_SMOKE" "$PROJECT/scripts/slurm_03_eval.sh")
ANALYZE=$(sbatch --parsable --dependency="afterok:$EVAL" "$PROJECT/scripts/slurm_04_analyze.sh")

echo "collect_second=$COLLECT_B"
echo "train_smoke=$TRAIN_SMOKE"
echo "train=$TRAIN"
echo "eval_smoke=$EVAL_SMOKE"
echo "eval=$EVAL"
echo "analyze=$ANALYZE"
