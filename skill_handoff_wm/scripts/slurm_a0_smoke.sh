#!/usr/bin/env bash
#SBATCH --job-name=handoff_a0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/handoff_a0_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/skill_handoff_wm"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
DATA=/mnt/data/vhoangth2/datasets/ogbench_data
OUT="$PROJECT/outputs/a0_smoke/${SLURM_JOB_ID}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/data.py" \
  "$PROJECT/policy.py" \
  "$PROJECT/sim.py" \
  "$PROJECT/scripts/train_gcbc.py" \
  "$PROJECT/scripts/eval_a0.py" \
  "$PROJECT/scripts/slurm_a0_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/train_gcbc.py" \
  --train-data "$DATA/antmaze-medium-navigate-v0.npz" \
  --val-data "$DATA/antmaze-medium-navigate-v0-val.npz" \
  --out-dir "$OUT/train" \
  --seed 0 --steps 20000 --batch-size 1024 --log-every 2000 --val-batches 10

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_a0.py" \
  --dataset-name antmaze-medium-navigate-v0 \
  --dataset-dir "$DATA" \
  --val-data "$DATA/antmaze-medium-navigate-v0-val.npz" \
  --checkpoint "$OUT/train/best.pt" \
  --out-dir "$OUT/eval" \
  --horizons 10,20,40,80 --cases-per-horizon 10
