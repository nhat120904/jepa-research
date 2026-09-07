#!/usr/bin/env bash
#SBATCH --job-name=spwm_elite
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_elite_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh

ARM=${ARM:?submit with ARM=latent_l2 or a progress arm name}
PLAN_SEED=${PLAN_SEED:?submit with PLAN_SEED}
OFFSETS=${OFFSETS:-"100"}
MIX=${MIX:-0.00}
NUM_EPISODES=${NUM_EPISODES:-12}
EPISODE_SEED=${EPISODE_SEED:-90100}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
WM_RUN=${WM_RUN:-lewm_20260904}
WM_SEED=${WM_SEED:-0}
PROG_RUN=${PROG_RUN:-progress_20260904}
PROG_SEED=${PROG_SEED:-0}
GOAL_SCALE=${GOAL_SCALE:-1.0}
N_PER_STRATUM=${N_PER_STRATUM:-6}
MAX_INSTRUMENTED=${MAX_INSTRUMENTED:-3}
RUN_ID=${RUN_ID:-elite_optimism_20260904}

sha256sum "$PROJECT/elite_optimism.py" "$PROJECT/scene_eval.py" \
          "$PROJECT/scene_render.py" "$PROJECT/progress_objective.py" \
          "$PROJECT/scripts/eval_elite_optimism.py" \
          "$PROJECT/scripts/slurm_elite_optimism.sh" \
          "$REPO/event_smdp_h0/scripts/run_scene_gate0.py"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The ladder loads lewm_best.pt; this gate follows it so the instrument measures
# the checkpoint the ladder actually planned with, not a different one.
CKPT="$CACHE_ROOT/checkpoints/$WM_RUN/seed$WM_SEED/lewm_best.pt"
if [[ "$ARM" == "latent_l2" ]]; then
  OBJECTIVE_ARGS=(--objective latent_l2)
else
  OBJECTIVE_ARGS=(
    --objective progress
    --progress-checkpoint "$CACHE_ROOT/checkpoints/$PROG_RUN/$ARM/seed$PROG_SEED/progress_head.pt"
    --mixture-weight "$MIX"
    --goal-scale "$GOAL_SCALE"
  )
fi

for OFFSET in $OFFSETS; do
  echo "--- arm=$ARM offset=$OFFSET seed=$PLAN_SEED shard=$SHARD_INDEX/$NUM_SHARDS $(date -u +%FT%TZ)"
  "$PY" "$PROJECT/scripts/eval_elite_optimism.py" \
    --checkpoint "$CKPT" \
    --cache-dir "$CACHE_ROOT/cache/val" \
    --out-dir "$PROJECT/outputs/elite_optimism/diagnostic/$RUN_ID" \
    "${OBJECTIVE_ARGS[@]}" \
    --goal-offset "$OFFSET" \
    --num-episodes "$NUM_EPISODES" \
    --episode-seed "$EPISODE_SEED" \
    --plan-seed "$PLAN_SEED" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --n-per-stratum "$N_PER_STRATUM" \
    --max-instrumented-replans "$MAX_INSTRUMENTED"
done

echo "DONE $(date -u +%FT%TZ)"
