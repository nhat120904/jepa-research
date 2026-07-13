#!/usr/bin/env bash
#SBATCH --job-name=jepa_extract_mw
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/extract_jepa_mw_%j.out
# One-off: extract the missing metaworld__jepa_wm_metaworld.h5 latent cache.
# scripts/03_extract_latents.py only checks `cache_path.exists()` to skip a model —
# it does NOT verify completeness, so a timeout mid-write would leave a corrupt/
# partial .h5 that a later run would wrongly trust. Run this standalone first and
# confirm the "Wrote ..." success line before anything downstream reads the cache.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_ONLY_MODEL=jepa_wm_metaworld

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
.venv/bin/python scripts/03_extract_latents.py --config configs/diagnostic_metaworld.yaml
echo "===== EXTRACT_DONE ====="; date
ls -la data/precomputed_latents/metaworld__jepa_wm_metaworld.h5
