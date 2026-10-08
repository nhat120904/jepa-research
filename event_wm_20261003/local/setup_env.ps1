# Create / refresh the local venv for event_wm (idempotent).
#   .\local\setup_env.ps1            # venv + packages + check_env.py
# Needs: uv on PATH, an NVIDIA driver that supports CUDA 12.8 (RTX 50xx = Blackwell needs cu128 builds).
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\env.ps1"
Set-Location $Repo

if (-not (Test-Path $EventWmPy)) { uv venv .venv --python 3.10 }
# 1) torch from the cu128 index only (mixing in PyPI would pick the CPU wheel).
uv pip install --python $EventWmPy "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
# 2) the rest from PyPI.
uv pip install --python $EventWmPy -r "$PSScriptRoot\requirements.txt"
uv pip freeze --python $EventWmPy | Set-Content -Encoding utf8 "$PSScriptRoot\requirements.lock.txt"
& $EventWmPy "$PSScriptRoot\check_env.py"
