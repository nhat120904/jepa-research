#!/usr/bin/env bash
#SBATCH --job-name=jepa_trmeval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-33%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/trm_eval_%A_%a.out
#
# Locked fresh-seed comparison under the same MetaWorld oracle-dynamics harness:
#   0-1   true simulator-state oracle (model-independent)
#   2-9   L2 and robust stateprobe controls, two models x two tasks
#   10-33 TRM replacement/hybrid, two models x three head seeds x two tasks
#
# Suggested submission (from diagnosis/, after reviewing resource availability):
#   T=$(sbatch --parsable scripts/slurm_trm_train.sh)
#   sbatch --dependency=afterok:$T scripts/slurm_trm_eval.sh
# Do not submit from this wrapper and do not run its Python commands on login.
set -euo pipefail

cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
IDX=${SLURM_ARRAY_TASK_ID:?submit as array 0-33}
MODELS=(dino_wm_metaworld jepa_wm_metaworld)
TASKS=(mw-push mw-pick-place)
EPISODES=${TRM_EPISODES:-64}
SEED0=${TRM_SEED0:-30000}

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if (( IDX < 2 )); then
  TASK=${TASKS[$IDX]}
  OUT=results/trm_heldout_oracle_${TASK}_seed${SEED0}_n${EPISODES}.csv
  echo "TRM control=true_state_oracle task=$TASK seed0=$SEED0 n=$EPISODES out=$OUT"
  "$PY" scripts/29_oracle_ceiling.py --config "$CFG" --tasks "$TASK" \
    --episodes "$EPISODES" --seed0 "$SEED0" --strict-success --out "$OUT"
  exit 0
fi

if (( IDX < 10 )); then
  J=$((IDX - 2))
  MODEL=${MODELS[$((J / 4))]}
  REM=$((J % 4))
  COSTS=(l2 stateprobe)
  COST=${COSTS[$((REM / 2))]}
  TASK=${TASKS[$((REM % 2))]}
  OUT=results/trm_heldout_${MODEL}_${COST}_${TASK}_seed${SEED0}_n${EPISODES}.csv
  EXTRA=()
  if [[ "$COST" == stateprobe ]]; then
    OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
    EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt
    test -f "$OBJ"; test -f "$EEP"
    EXTRA+=(--probe "$OBJ" --ee-probe "$EEP")
  fi
  echo "TRM control=$COST model=$MODEL task=$TASK seed0=$SEED0 n=$EPISODES out=$OUT"
  "$PY" scripts/30_latent_oracle.py --config "$CFG" --model "$MODEL" \
    --cost "$COST" "${EXTRA[@]}" --tasks "$TASK" \
    --episodes "$EPISODES" --seed0 "$SEED0" --strict-success --out "$OUT"
  exit 0
fi

J=$((IDX - 10))
MODEL=${MODELS[$((J / 12))]}
REM=$((J % 12))
HEAD_SEED=$((REM / 4))
SUB=$((REM % 4))
MODES=(replacement hybrid)
MODE=${MODES[$((SUB / 2))]}
TASK=${TASKS[$((SUB % 2))]}
HEAD=checkpoints/trm_${MODEL}_s${HEAD_SEED}.pt
test -f "$HEAD"
OUT=results/trm_heldout_${MODEL}_${MODE}_h${HEAD_SEED}_${TASK}_seed${SEED0}_n${EPISODES}.csv
echo "TRM mode=$MODE model=$MODEL head_seed=$HEAD_SEED task=$TASK seed0=$SEED0 n=$EPISODES out=$OUT"
"$PY" scripts/51_trm_oracle_eval.py --config "$CFG" --model "$MODEL" \
  --trm-head "$HEAD" --mode "$MODE" --hybrid-weight "${TRM_HYBRID_WEIGHT:-1.0}" \
  --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
  --strict-success --out "$OUT"

echo "===== TRM_EVAL_DONE index=$IDX ===== $(date)"
