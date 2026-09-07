#!/usr/bin/env bash
#SBATCH --job-name=mwm_g4_eval
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g4_eval_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/evaluate_kernel_mmr.py" \
          "$PROJECT/scripts/slurm_gate4_evaluate.sh"
"$PY" "$PROJECT/scripts/evaluate_kernel_mmr.py" \
  --features "$PROJECT/outputs/gate3_full/features.pt" \
  --models-root "$PROJECT/outputs/gate4_full/models" \
  --selection "$PROJECT/outputs/gate4_full/selection.json" \
  --baseline-summary "$PROJECT/outputs/gate3_full/evaluation/summary.json" \
  --baseline-actions "$PROJECT/outputs/gate3_full/evaluation/planned_actions.pt" \
  --gate1-actions "$PROJECT/outputs/gate1_full/actions.pt" \
  --out-dir "$PROJECT/outputs/gate4_full/evaluation" --seeds 0 1 2 \
  --num-samples 256 --iterations 6 --restarts 2 --bootstrap 20000 --device cuda
