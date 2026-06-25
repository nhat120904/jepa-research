#!/usr/bin/env bash
#SBATCH --job-name=jepa_reachS
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/reach_strict_%j.out
# D.2 — Reach success-criterion STRICT re-score (end-of-episode judging).
# The headline "94% vs paper 44.8%" used the any-step success latch (TD-MPC2
# convention); the paper judges at episode end. --strict-success runs the full
# episode and reports the env flag AT THE FINAL STEP ('success_end'), while still
# recording the any-step latch ('success') in the same CSV for a paired delta.
# Same protocol/seeds as results/closed_loop_report.md (pooled probe, beta=0.1),
# so this directly re-judges the SAME Reach episodes the 94% came from.
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
M=dino_wm_metaworld
PROBE=checkpoints/object_probe_${M}.pt          # pooled probe — matches the original 94% run
DYN=checkpoints/object_dynamics_${M}.pt
OUT=${RS_OUT:-results/metaworld_reach_strict.csv}
EPISODES=${RS_EPISODES:-16}
TASKS=${RS_TASKS:-"mw-reach"}
ARMS=${RS_ARMS:-"l2 hdyn"}

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "STRICT episode-end re-score  arms=[$ARMS]  tasks=[$TASKS]  episodes=$EPISODES  out=$OUT"
ls -la "$PROBE" "$DYN"

set -e
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --probe "$PROBE" --dyn-head "$DYN" --beta 0.1 \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" \
    --strict-success --out "$OUT"
echo "===== REACH_STRICT_DONE ====="; date
cat "$OUT"
