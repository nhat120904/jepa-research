# Scene memory v2 on this PC: sm2_train.py stages (agent -> label -> seg -> segpred -> scene -> stargets -> seethru learned /
# pixel) then sm2_diag.py on VAL.
#   .\local\run_sm2.ps1 -Family scene -DataEnv visual-scene-play-v0
#   .\local\run_sm2.ps1 -Family scene -DataEnv visual-scene-play-v0 -Out <existing run dir>     # continue: finished stages are skipped
# Same hyper-parameters for every family (sm2_train.py defaults). Stages with resume.pt (agent, seg, scene) continue after a
# crash of the GPU python (seen on this PC); a stage is relaunched up to -MaxRetries times.
param(
    [Parameter(Mandatory = $true)][ValidateSet('cube', 'puzzle', 'scene')][string]$Family,
    [Parameter(Mandatory = $true)][string]$DataEnv,
    [int]$Episodes = 1000,
    [string]$Out = '',
    [string]$DiagName = 'diag',
    [int]$MaxRetries = 4
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1" | Out-Null
$Repo = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $env:EVENT_WM_RUNS 'cache'
$cache = Join-Path $cacheRoot $DataEnv
if (-not $Out) { $Out = Join-Path $env:EVENT_WM_RUNS ("scene_memory_v2\{0}_{1}" -f $Family, (Get-Date -Format 'yyyyMMddHHmmss')) }
New-Item -ItemType Directory -Force "$Out\src" | Out-Null
Get-ChildItem "$Repo\scripts\sm2_*.py", "$Repo\scripts\sm_model.py", "$Repo\scripts\sm_diag.py", "$PSScriptRoot\run_sm2.ps1" | Get-FileHash -Algorithm SHA256 |
    ForEach-Object { "{0}  {1}" -f $_.Hash.ToLower(), (Split-Path $_.Path -Leaf) } | Set-Content "$Out\src\SOURCE_SHA256SUMS"
"RUN DIR: $Out"

if (-not (Test-Path "$cache\cache_info.json")) {
    "=== cache $DataEnv ($Episodes train episodes)"
    $env:SLURM_JOB_ID = 'localcache'
    & $EventWmPy "$Repo\scripts\cache_data.py" --data $env:EVENT_WM_DATA --env $DataEnv --train-episodes $Episodes --out $cacheRoot
    if ($LASTEXITCODE -ne 0) { throw 'cache_data failed' }
}

# PYTHONPATH root only supplies a run root to run_stage; the sm2_* scripts import nothing from it.
$root = Join-Path $env:EVENT_WM_RUNS 'unified_local'
$common = @('--cache', $cache, '--out', "$Out\train", '--episodes', $Episodes, '--resume')
$stages = @(
    @{ name = 'agent'; done = "$Out\train\agent_net.pt" },
    @{ name = 'label'; done = "$Out\train\teacher.json" },
    @{ name = 'seg'; done = "$Out\train\segmenter.pt" },
    @{ name = 'segpred'; done = "$Out\train\seg_val.npy" },
    @{ name = 'scene'; done = "$Out\train\scene.pt" },
    @{ name = 'stargets'; done = "$Out\train\see_targets.json" },
    @{ name = 'seethru'; space = 'learned'; done = "$Out\train\seethru_learned.pt" },
    @{ name = 'seethru'; space = 'pixel'; done = "$Out\train\seethru_pixel.pt" }
)
foreach ($s in $stages) {
    if (Test-Path $s.done) { "=== $($s.name): done, skipped"; continue }
    $ok = $false
    for ($try = 1; $try -le $MaxRetries -and -not $ok; $try++) {
        "=== $($s.name) attempt $try $(Get-Date -Format s)"
        $extra = @(); if ($s.space) { $extra = @('--space', $s.space) }
        & "$PSScriptRoot\run_stage.ps1" -Root $root -Script "$Repo\scripts\sm2_train.py" -Name "sm2_${Family}_$($s.name)$($s.space)" -ScriptArgs (@('--stage', $s.name) + $common + $extra)
        if ($LASTEXITCODE -eq 0 -and (Test-Path $s.done)) { $ok = $true } else { "attempt $try failed (exit $LASTEXITCODE)"; Start-Sleep -Seconds 20 }
    }
    if (-not $ok) { throw "stage $($s.name) did not finish after $MaxRetries attempts; see $env:EVENT_WM_RUNS\logs" }
}

"=== diag $(Get-Date -Format s)"
& "$PSScriptRoot\run_stage.ps1" -Root $root -Script "$Repo\scripts\sm2_diag.py" -Name "sm2_${Family}_diag" `
    -ScriptArgs @('--run', "$Out\train", '--cache', $cache, '--family', $Family, '--out', "$Out\$DiagName")
if ($LASTEXITCODE -ne 0) { throw "sm2_diag failed (exit $LASTEXITCODE)" }
"SM2_DONE $Out"
