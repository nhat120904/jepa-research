#!/usr/bin/env bash
#SBATCH --job-name=jepa_precladder
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/precision_ladder_%j.out
# In-sim analogue of V-JEPA-2 cup-vs-box grasping (Table 2: cup 65%, box 25% on a
# real Franka). The exact numbers need a physical robot; the MECHANISM — success
# degrading as the task demands finer object precision — is reproduced here by
# running the L2 baseline closed-loop over a precision-difficulty ladder and
# sweeping the object tolerance. Same protocol as scripts/18/29/30 (strict).
#   reach (free-space anchor) -> push -> pick-place -> peg-insert-side (precision ceiling)
set -uo pipefail
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
M=${PL_MODEL:-dino_wm_metaworld}
PROBE=checkpoints/object_probe_${M}.pt
DYN=checkpoints/object_dynamics_${M}.pt
OUT=${PL_OUT:-results/metaworld_precision_ladder.csv}
EPISODES=${PL_EPISODES:-16}
TASKS=${PL_TASKS:-"mw-reach mw-push mw-pick-place mw-peg-insert-side"}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M tasks=[$TASKS] episodes=$EPISODES out=$OUT"
ls -la "$PROBE" "$DYN"

set -e
# 1) closed-loop L2 baseline over the ladder (probe/dyn-head are required args but
#    unused by the l2 arm; pass the existing ckpts).
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" --probe "$PROBE" --dyn-head "$DYN" \
    --tasks $TASKS --arms l2 --episodes "$EPISODES" --out "$OUT" --strict-success

# 2) build the precision-ladder figure + summary (env-flag success + tolerance sweep)
$PY scripts/32_precision_ladder.py --csv "$OUT" --arm l2
echo "===== PRECISION_LADDER_DONE ====="; date
cat results/metaworld_precision_ladder_summary.md
