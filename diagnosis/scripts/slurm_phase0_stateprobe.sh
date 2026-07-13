#!/usr/bin/env bash
#SBATCH --job-name=jepa_p0sp
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase0_stateprobe_%j.out
# Phase-0 gate, CORRECTED. The plain gobj gate (job 23341) failed push/pick 0/8 under
# perfect dynamics, but Test-1b (scripts/21) showed the spatial probe localises the
# static object <5cm 92% of the time -> NOT a readout-precision wall. The likely
# confound: gobj omitted the hand-APPROACH term that the state-oracle cost has
# (scripts/29: ‖obj-goal‖ + w_hand·‖hand-obj‖). This run uses --cost stateprobe = the
# EXACT state-oracle cost but with object+hand read from PROBES (spatial + ee), under
# perfect latent dynamics. It isolates the readout from the cost form:
#   stateprobe flips push/pick 0 -> >0  => the cost lever is ALIVE (state-oracle's win
#                                          survives the probe swap) -> Track A/B viable.
#   stateprobe stays 0 (while state-oracle true-state = 16/16, 11/16)
#                                       => the probe readout, used IN THE PLANNING LOOP,
#                                          is the wall after all -> Phase 3 (representation).
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
M=${P0_MODEL:-dino_wm_metaworld}
PROBE=${P0_PROBE:-checkpoints/spatial_object_probe_${M}.pt}
EEPROBE=${P0_EEPROBE:-checkpoints/ee_probe_${M}.pt}
WHAND=${P0_WHAND:-0.5}
EPISODES=${P0_EPISODES:-16}
TASKS=${P0_TASKS:-"mw-push mw-pick-place"}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M probe=$PROBE ee=$EEPROBE w_hand=$WHAND tasks=[$TASKS] episodes=$EPISODES"
ls -la "$PROBE" "$EEPROBE"

set -e
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost stateprobe --probe "$PROBE" --ee-probe "$EEPROBE" --w-hand "$WHAND" \
    --tasks $TASKS --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_stateprobe.csv
echo "===== PHASE0_STATEPROBE_DONE ====="; date
cat results/latent_oracle_stateprobe.csv
