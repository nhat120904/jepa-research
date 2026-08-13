#!/usr/bin/env bash
#SBATCH --job-name=jepa_budgettest
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/budget_test_%A_%a.out
#
# Diagnostic only (not headline evidence): does the oracle's 0/16-12.5% on
# door-open/box-close/shelf-place come from the 100-step budget copied from
# push/pick-place, rather than the task being unsolvable under this cost?
# expert_success_step data already shows the scripted expert itself barely
# finishes (door-open) or mostly never finishes (box-close, 10/16 seeds) in
# 100 steps -> Gate-1 hypothesis. Oracle-only (cheap, no encoder); same seed0
# as the original cell for direct before/after comparison. assembly/lever-pull
# excluded: their oracle already matches goal xyz to ~1-2mm and still fails,
# so more budget cannot fix a cost-specification mismatch.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
EPISODES=16
BUDGET=300
IDX=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}

TASKS=(mw-door-open mw-box-close mw-shelf-place)
SEEDS=(70000 71000 71000)

TASK=${TASKS[$IDX]}
SEED0=${SEEDS[$IDX]}
OUT=results/budget_test_${TASK}_oracle_seed${SEED0}_budget${BUDGET}.csv

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "budget test: task=$TASK seed0=$SEED0 budget=$BUDGET out=$OUT"
echo "commit=$(git rev-parse HEAD)"

"$PY" scripts/29_oracle_ceiling.py --config "$CFG" \
  --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
  --max-episode-steps "$BUDGET" --strict-success --out "$OUT"

echo "===== BUDGET_TEST_CELL_DONE ===== $(date)"
