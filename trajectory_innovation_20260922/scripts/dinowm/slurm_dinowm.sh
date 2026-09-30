#!/usr/bin/env bash
#SBATCH --job-name=ti_dinowm
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/dinowm_%x_%j.out
# DINO-WM (ICML 2025) PushT planning baseline for CTA (docs/CTA_DINOWM_PLANNING_PROTOCOL.md).
#   repro  official checkpoint + official plan_pusht.yaml (MPC-CEM, H=5, 300 samples, 30 iters, goal_H=5,
#          50 evals, seed 99); max_iter capped at 5 (config default null loops until every eval succeeds).
#          MPC iter 0 = open-loop CEM over the full horizon (n_taken_actions = horizon = 5).
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
DINO=/mnt/data/nhatnc129/jepa/dino_wm_original
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
EXTRA=/mnt/data/nhatnc129/jepa/trajectory_innovation/dinowm_extra_site
OUT=/mnt/data/nhatnc129/jepa/trajectory_innovation/dinowm_${MODE}_${SLURM_JOB_ID}
export DATASET_DIR=/mnt/data/nhatnc129/jepa/datasets WANDB_MODE=disabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
git -C "$DINO" rev-parse HEAD > "$OUT/dino_wm_commit.txt" 2>/dev/null || echo "not a git checkout" > "$OUT/dino_wm_commit.txt"
# Pinned to the authors' environment.yaml where the CTA venv differs: gym 0.23.1 (old API), pymunk 6.8.0
# (pymunk 7 removed add_collision_handler, used by env/pusht), decord (video dataset), psutil (utils.py), accelerate 0.26.1 (plan.py imports train.py), scikit-image 0.25.2 (env/pusht; 0.22 wheels are numpy-1 ABI, the venv has numpy 2.2.6); full import scan of the PushT path done 09-27.
# --no-deps keeps the CTA venv's numpy/torch; the extra dir precedes site-packages on PYTHONPATH.
if [[ ! -f "$EXTRA/.pinned_v5" ]]; then
  ~/.local/bin/uv pip install --python "$PY" --target "$EXTRA" --no-deps --link-mode=copy \
    decord==0.6.0 gym==0.23.1 gym-notices==0.0.8 pymunk==6.8.0 psutil accelerate==0.26.1 scikit-image==0.25.2 tifffile lazy_loader
  touch "$EXTRA/.pinned_v5"
fi
export PYTHONPATH="$EXTRA"
case "$MODE" in
  repro)
    cd "$OUT"
    START=$(date +%s)
    set +e
    "$PY" "$PROJECT/scripts/dinowm/plan_official.py" "$DINO" --config-name plan_pusht.yaml \
      model_name=pusht ckpt_base_path=/mnt/data/nhatnc129/jepa/dino_wm_official planner.max_iter=5 \
      hydra.run.dir="$OUT/run" > "$OUT/plan.log" 2>&1
    RC=$?
    set -e
    echo "exit=$RC wall_seconds=$(( $(date +%s) - START ))" | tee "$OUT/exit.txt"
    grep -E "MPC iter|Success rate|Error|error" "$OUT/plan.log" | tail -40 || true
    exit $RC
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
