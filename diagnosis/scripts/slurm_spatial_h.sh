#!/usr/bin/env bash
#SBATCH --job-name=jepa_sph
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spatial_h_%j.out
# Spatial-h planning leg on the cluster (SLURM port of run_spatial_h_sweep.ps1).
# Plan against the dyn-head object trajectory with the SPATIAL object probe (2cm
# aim) and an object-dominant cost (beta high). l2 baseline vs hdyn(spatial),
# paired, on the contact tasks. scripts/18 resumes its own CSV, so re-submitting
# this job continues an interrupted sweep.
#
# Closed-loop Metaworld renders the env headlessly -> MUJOCO_GL=egl (validated by
# the smoke first; see SPH_SMOKE=1 below).
#
# Knobs (env overrides):
#   SPH_SMOKE=1   -> tiny validation run (1 ep, 30 samples, 3 iters, push only)
#   SPH_EPISODES, SPH_SAMPLES, SPH_ITERS, SPH_TASKS, SPH_ARMS, SPH_BETA, SPH_OUT
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl                       # headless GPU rendering on the compute node
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml

PROBE=checkpoints/spatial_object_probe_dino_wm_metaworld.pt
DYN=checkpoints/object_dynamics_dino_wm_metaworld.pt

if [[ "${SPH_SMOKE:-0}" == "1" ]]; then
  EPISODES=${SPH_EPISODES:-1}; SAMPLES=${SPH_SAMPLES:-30}; ITERS=${SPH_ITERS:-3}
  TASKS=${SPH_TASKS:-"mw-push"}; ARMS=${SPH_ARMS:-"l2 hdyn"}; BETA=${SPH_BETA:-5.0}
  OUT=${SPH_OUT:-results/metaworld_spatial_h_smoke.csv}
else
  EPISODES=${SPH_EPISODES:-16}; SAMPLES=${SPH_SAMPLES:-300}; ITERS=${SPH_ITERS:-15}
  TASKS=${SPH_TASKS:-"mw-push mw-pick-place"}; ARMS=${SPH_ARMS:-"l2 hdyn"}; BETA=${SPH_BETA:-5.0}
  OUT=${SPH_OUT:-results/metaworld_spatial_h.csv}
fi

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "MUJOCO_GL=$MUJOCO_GL  probe=$PROBE  dyn=$DYN"
echo "episodes=$EPISODES samples=$SAMPLES iters=$ITERS beta=$BETA tasks=[$TASKS] arms=[$ARMS] out=$OUT"
ls -la "$PROBE" "$DYN"

set -e
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model dino_wm_metaworld \
    --probe "$PROBE" --dyn-head "$DYN" --beta "$BETA" \
    --tasks $TASKS --arms $ARMS \
    --episodes "$EPISODES" \
    --cem-num-samples "$SAMPLES" --cem-iterations "$ITERS" \
    --out "$OUT"
echo "===== SPATIAL_H_DONE ====="; date
ls -la "$OUT"
