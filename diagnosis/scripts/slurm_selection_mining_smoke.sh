#!/usr/bin/env bash
#SBATCH --job-name=jepa_sel_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/selection_smoke_%j.out
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

echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python scripts/59_mine_selection_populations.py \
  --config configs/diagnostic_metaworld.yaml \
  --model dino_wm_metaworld --task mw-push \
  --episodes 1 --seed0 61999 --max-episode-steps 3 \
  --cem-num-samples 8 --cem-iterations 1 \
  --top-proxy 3 --top-true 3 --random-count 3 \
  --probe checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
  --ee-probe checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
  --out results/selection_populations_smoke.pt
.venv/bin/python - <<'PY'
import torch
b = torch.load("results/selection_populations_smoke.pt", map_location="cpu", weights_only=False)
assert b["metadata"]["n_groups"] == 1
assert b["frames"].shape[0] == b["true_cost"].numel()
assert b["role_proxy"].any() and b["role_true"].any()
print("SELECTION_MINING_SMOKE_OK", b["metadata"])
PY
.venv/bin/python scripts/60_train_selection_encoder.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --buffer results/selection_populations_smoke.pt \
  --adaptation last_blocks --objective tail --seed 999 \
  --train-seeds 61999 --val-seeds 61999 --epochs 1 --steps-per-epoch 1 \
  --val-groups 1 --out checkpoints/selection_smoke.pt
.venv/bin/python scripts/61_eval_selection_encoder.py \
  --config configs/diagnostic_metaworld.yaml \
  --checkpoint checkpoints/selection_smoke.pt \
  --tasks mw-push mw-reach --episodes 1 --seed0 61999 \
  --max-episode-steps 3 --cem-num-samples 8 --cem-iterations 1 \
  --out results/selection_eval_smoke.csv
echo "SELECTION_END_TO_END_SMOKE_OK $(date)"
