#!/usr/bin/env bash
#SBATCH --job-name=ti_libero_l1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/libero_l1_%A_%a.out
# LIBERO L1 (docs/LIBERO_QUALIFICATION_PROTOCOL.md): P0 reproduction with the UNMODIFIED lerobot-eval
# pipeline. libero_goal, init states 0-9 (batch_size = n_episodes = 10). Array task i: seed i//2, tasks 0-4 (i even) or 5-9 (i odd).
# Split because one seed over all 10 tasks takes ~90 min (job 54429) and lerobot-eval only writes results at the end.
# HuggingFaceVLA/smolvla_libero pinned at 6721902, n_action_steps=10 (lerobot#4614).
# Hub stays online: the checkpoint names HuggingFaceTB/SmolVLM2-500M-Instruct, which the Hub redirects
# to SmolVLM2-500M-Video-Instruct (already cached at 7b375e1); offline mode cannot follow the redirect.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
ROOT=/mnt/data/nhatnc129/jepa/trajectory_innovation
SETUP="$ROOT/libero_l0_54414"
VENV="$SETUP/venv"
RUN="$ROOT/libero_l1_${SLURM_ARRAY_JOB_ID}"
SEED=$((SLURM_ARRAY_TASK_ID / 2))
HALF=$((SLURM_ARRAY_TASK_ID % 2))
if (( HALF == 0 )); then TASK_IDS="[0,1,2,3,4]"; else TASK_IDS="[5,6,7,8,9]"; fi
OUT="$RUN/seed_${SEED}_part${HALF}"
mkdir -p "$RUN"
[[ ! -e "$OUT" ]] || exit 2
if [[ ! -e "$RUN/CODE_SHA256SUMS" ]]; then
  cp -r "$PROJECT/scripts/libero" "$PROJECT/docs" "$RUN/" 2>/dev/null || true
  find "$RUN/libero" "$RUN/docs" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN/CODE_SHA256SUMS" || true
fi
CKPT=$("$VENV/bin/python" -c "import json;print(json.load(open('$SETUP/checkpoints.json'))['HuggingFaceVLA/smolvla_libero']['path'])")
export HF_HOME="$SETUP/hf_cache" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 LIBERO_CONFIG_PATH="$SETUP/l1_libero_config"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -e "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  "$VENV/bin/python" - <<PY
import importlib.util
from pathlib import Path
pkg = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
cfg = Path("$LIBERO_CONFIG_PATH"); cfg.mkdir(parents=True, exist_ok=True)
paths = {"benchmark_root": pkg, "bddl_files": pkg / "bddl_files", "init_states": pkg / "init_files",
         "datasets": cfg / "datasets", "assets": pkg / "assets"}
(cfg / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
PY
fi
nvidia-smi -L
"$VENV/bin/lerobot-eval" \
  --policy.path="$CKPT" \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --seed="$SEED" \
  --output_dir="$OUT" < /dev/null
