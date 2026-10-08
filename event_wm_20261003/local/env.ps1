# Dot-source from PowerShell:  . .\local\env.ps1
# Local equivalents of slurm/env.sh. Everything heavy lives under E:\jepa-data (outside git, off the small C: drive).
$Repo = Split-Path -Parent $PSScriptRoot
$script:EventWmPy = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not $env:EVENT_WM_DATA) { $env:EVENT_WM_DATA = 'E:\jepa-data\ogbench\data' }   # OGBench *.npz (fetch_data.py)
if (-not $env:EVENT_WM_RUNS) { $env:EVENT_WM_RUNS = 'E:\jepa-data\event_wm' }        # run roots, caches, logs
if (-not $env:HF_HOME)       { $env:HF_HOME = 'E:\jepa-data\hf_cache' }
$env:UV_LINK_MODE = 'copy'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
# MUJOCO_GL is left unset on purpose: Windows renders with GLFW; the cluster's osmesa/egl do not exist here.
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
Remove-Item Env:PYOPENGL_PLATFORM -ErrorAction SilentlyContinue
# The repo's scripts refuse to run numerical work unless SLURM_JOB_ID is set (guard against the cluster login node).
# This PC is not a login node, so a per-session "local" id satisfies the guard; run_stage.ps1 makes a fresh one per stage.
if (-not $env:SLURM_JOB_ID) { $env:SLURM_JOB_ID = 'local' + (Get-Date -Format 'yyyyMMddHHmmss') }
New-Item -ItemType Directory -Force $env:EVENT_WM_RUNS, "$env:EVENT_WM_RUNS\logs" | Out-Null
Write-Host "event_wm env: py=$EventWmPy data=$env:EVENT_WM_DATA runs=$env:EVENT_WM_RUNS"
