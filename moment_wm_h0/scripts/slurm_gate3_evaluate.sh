#!/usr/bin/env bash
#SBATCH --job-name=mwm_g3_eval
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g3_eval_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/evaluate_strong_baselines.py" \
          "$PROJECT/scripts/slurm_gate3_evaluate.sh"
"$PY" "$PROJECT/scripts/evaluate_strong_baselines.py" \
  --features "$PROJECT/outputs/gate3_full/features.pt" \
  --checkpoints-root "$PROJECT/outputs/gate3_full/models" \
  --gate1-actions "$PROJECT/outputs/gate1_full/actions.pt" \
  --out-dir "$PROJECT/outputs/gate3_full/evaluation" \
  --arms mse cov_mse cadm --seeds 0 1 2 \
  --num-samples 256 --iterations 6 --restarts 2 \
  --bootstrap 20000 --device cuda
