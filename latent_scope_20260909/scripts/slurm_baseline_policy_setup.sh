#!/usr/bin/env bash
#SBATCH --job-name=lscope_base_setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/setup_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
POLICY_VENV="$RUNTIME_ROOT/policy_venv"
SIM_VENV="$RUNTIME_ROOT/sim_venv"
GROOT_SRC="$RUNTIME_ROOT/src/Isaac-GR00T"
ROBOCASA_SRC="$SHARED_ROOT/src/robocasa365"
ROBOSUITE_SRC="$SHARED_ROOT/src/robosuite"
GROOT_REVISION=9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
UV_BIN=/home/nhatnc129/.local/bin/uv

mkdir -p "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/outputs" "$RUNTIME_ROOT/src" \
    "$RUNTIME_ROOT/checkpoints" "$RUNTIME_ROOT/hf_cache"

if [[ ! -d "$GROOT_SRC/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://github.com/robocasa-benchmark/Isaac-GR00T.git "$GROOT_SRC"
fi

if [[ -n "$(git -C "$GROOT_SRC" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to alter a dirty Isaac-GR00T checkout: $GROOT_SRC" >&2
    exit 1
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT_SRC" fetch origin "$GROOT_REVISION"
GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT_SRC" checkout --detach "$GROOT_REVISION"

if [[ ! -x "$POLICY_VENV/bin/python" ]]; then
    "$UV_BIN" venv "$POLICY_VENV" --python 3.10
fi

if ! PYTHONPATH="$GROOT_SRC" "$POLICY_VENV/bin/python" -c \
    'import diffusers, gr00t, pytorch3d, torch, torchvision; assert torch.__version__.startswith("2.5.1"); assert torchvision.__version__.startswith("0.20.1")'; then
    "$UV_BIN" pip install --python "$POLICY_VENV/bin/python" -e "$GROOT_SRC[base]"
fi

if ! "$POLICY_VENV/bin/python" -c 'import flash_attn'; then
    TORCH_ABI=$(
        "$POLICY_VENV/bin/python" -c \
            'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")'
    )
    FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abi${TORCH_ABI}-cp310-cp310-linux_x86_64.whl"
    "$UV_BIN" pip install --python "$POLICY_VENV/bin/python" "$FLASH_WHEEL"
fi

if [[ ! -x "$SIM_VENV/bin/python" ]]; then
    "$UV_BIN" venv "$SIM_VENV" --python 3.11
fi

if ! "$SIM_VENV/bin/python" -c \
    'import gymnasium, robocasa, robosuite, torch, zmq; assert gymnasium.__version__ == "0.29.1"'; then
    "$UV_BIN" pip install --python "$SIM_VENV/bin/python" \
        "gymnasium==0.29.1" pyzmq -e "$ROBOSUITE_SRC" -e "$ROBOCASA_SRC"
fi

export HF_HOME="$RUNTIME_ROOT/hf_cache"
export PYTHONPATH="$GROOT_SRC:${PYTHONPATH:-}"
"$POLICY_VENV/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/prepare_baseline_policy.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/baseline_policy_gate.json" \
    --sim-python "$SIM_VENV/bin/python" \
    --output "$RUNTIME_ROOT/outputs/setup_result.json"
