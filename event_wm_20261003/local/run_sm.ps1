# Scene-memory v1 at cluster scale on this PC: cache -> sm_train (self-healing) -> sm_diag. Local stand-in for slurm/sm_pipeline.sh.
#   .\local\run_sm.ps1 -Family scene -DataEnv visual-scene-play-v0
#   .\local\run_sm.ps1 -Family scene -DataEnv visual-scene-play-v0 -Out <existing run dir>     # continue after an interruption
#   add -GateSt for the gate fix (sm_train --gate-st); pass the same switches again when continuing a run
# Same hyper-parameters as the cluster (STEPS 20000, EPISODES 1000, batch 16, clip 32). Only how the data is read changes
# (--mmap: frames stay on disk, the cluster copied 12 GB into RAM) plus a resume checkpoint every 1000 steps. A crash of the
# GPU python (seen on this PC) relaunches with --resume, up to -MaxRetries times.
param(
    [Parameter(Mandatory = $true)][ValidateSet('cube', 'puzzle', 'scene')][string]$Family,
    [Parameter(Mandatory = $true)][string]$DataEnv,
    [int]$Steps = 20000,
    [int]$Episodes = 1000,
    [int]$ExportEpisodes = -1,
    [int]$Accum = 2,   # micro-batches per batch of 16: one pass of 16 clips needs 12.6 GiB and spills out of this card's 11.9 GiB (4x slower)
    [string]$Out = '',
    [int]$MaxRetries = 8,
    [switch]$GateSt,   # sm_train --gate-st: closed gates get the reconstruction gradient (v1 gates all closed by step 500, never reopened)
    [double]$WMem = 0,  # sm_train --w-mem: memory-only reconstruction weight
    [ValidateSet('sigmoid', 'softplus')][string]$GateL0 = 'sigmoid'   # sm_train --gate-l0: L0 surrogate
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1" | Out-Null
$Repo = Split-Path -Parent $PSScriptRoot
if ($ExportEpisodes -lt 0) { $ExportEpisodes = $Episodes }
$cacheRoot = Join-Path $env:EVENT_WM_RUNS 'cache'
$cache = Join-Path $cacheRoot $DataEnv
if (-not $Out) { $Out = Join-Path $env:EVENT_WM_RUNS ("scene_memory_v1\{0}_{1}" -f $Family, (Get-Date -Format 'yyyyMMddHHmmss')) }
New-Item -ItemType Directory -Force "$Out\src" | Out-Null
Get-ChildItem "$Repo\scripts\sm_*.py", "$PSScriptRoot\*.ps1" | Get-FileHash -Algorithm SHA256 |
    ForEach-Object { "{0}  {1}" -f $_.Hash.ToLower(), (Split-Path $_.Path -Leaf) } | Set-Content "$Out\src\SOURCE_SHA256SUMS"
"RUN DIR: $Out"

if (-not (Test-Path "$cache\cache_info.json")) {
    "=== cache $DataEnv ($Episodes train episodes)"
    $env:SLURM_JOB_ID = 'localcache'
    & $EventWmPy "$Repo\scripts\cache_data.py" --data $env:EVENT_WM_DATA --env $DataEnv --train-episodes $Episodes --out $cacheRoot
    if ($LASTEXITCODE -ne 0) { throw 'cache_data failed' }
}

# PYTHONPATH root only supplies a run root to run_stage; the sm_* scripts import nothing from it.
$root = Join-Path $env:EVENT_WM_RUNS 'unified_local'
$trainArgs = @('--cache', $cache, '--episodes', $Episodes, '--steps', $Steps, '--mmap', '--accum', $Accum, '--ckpt-every', 1000, '--resume', '--out', "$Out\train")
if ($GateSt) { $trainArgs += '--gate-st' }
if ($WMem -gt 0) { $trainArgs += @('--w-mem', $WMem) }
if ($GateL0 -ne 'sigmoid') { $trainArgs += @('--gate-l0', $GateL0) }
$done = $false
for ($try = 1; $try -le $MaxRetries -and -not $done; $try++) {
    "=== train attempt $try $(Get-Date -Format s)"
    & "$PSScriptRoot\run_stage.ps1" -Root $root -Script "$Repo\scripts\sm_train.py" -Name "sm_${Family}_train" -ScriptArgs $trainArgs
    if ($LASTEXITCODE -eq 0) { $done = $true } else { "attempt $try failed (exit $LASTEXITCODE); relaunching with --resume"; Start-Sleep -Seconds 20 }
}
if (-not $done) { throw "sm_train did not finish after $MaxRetries attempts; see $env:EVENT_WM_RUNS\logs" }

"=== diag $(Get-Date -Format s)"
& "$PSScriptRoot\run_stage.ps1" -Root $root -Script "$Repo\scripts\sm_diag.py" -Name "sm_${Family}_diag" `
    -ScriptArgs @('--model', "$Out\train\sm.pt", '--cache', $cache, '--family', $Family, '--episodes', $ExportEpisodes, '--out', "$Out\diag")
if ($LASTEXITCODE -ne 0) { throw "sm_diag failed (exit $LASTEXITCODE)" }
"SM_DONE $Out"

