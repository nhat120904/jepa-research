#!/usr/bin/env bash
#SBATCH --job-name=mwm_g4_smoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g4_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT="$PROJECT/outputs/gate4_smoke"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/train_kernel_mmr.py" \
          "$PROJECT/scripts/select_mmr_lambda.py" \
          "$PROJECT/scripts/evaluate_kernel_mmr.py" \
          "$PROJECT/scripts/slurm_gate4_smoke.sh"
"$PY" "$PROJECT/scripts/train_kernel_mmr.py" \
  --features "$PROJECT/outputs/gate3_smoke/features.pt" \
  --mmr-lambda 1 --seed 0 --out-dir "$ROOT/models/lambda_1/seed_0" \
  --epochs 2 --patience 2 --batch-size 64 --device cuda
"$PY" "$PROJECT/scripts/select_mmr_lambda.py" \
  --models-root "$ROOT/models" --lambdas 1 --seeds 0 \
  --out "$ROOT/selection.json"
"$PY" "$PROJECT/scripts/evaluate_kernel_mmr.py" \
  --features "$PROJECT/outputs/gate3_smoke/features.pt" \
  --models-root "$ROOT/models" --selection "$ROOT/selection.json" \
  --baseline-summary "$PROJECT/outputs/gate3_smoke/evaluation/summary.json" \
  --baseline-actions "$PROJECT/outputs/gate3_smoke/evaluation/planned_actions.pt" \
  --gate1-actions "$PROJECT/outputs/gate1_full/actions.pt" \
  --out-dir "$ROOT/evaluation" --seeds 0 \
  --num-samples 64 --iterations 2 --restarts 1 --bootstrap 200 --device cuda
