# End-to-end smoke test of the STATE pipeline on one family (default cube): tiny WM/h, skill, support, 1 closed-loop episode.
# Checks that every stage runs on this GPU and reports step speed. The resulting models are NOT meaningful.
#   .\local\smoke.ps1 [-Root E:\jepa-data\event_wm\unified_local] [-Family cube] [-Steps 300]
# Needs run_prepare.py to have finished for the root. Writes only to <Root>\smoke_<timestamp>; never touches real runs.
param(
    [string]$Root = 'E:\jepa-data\event_wm\unified_local',
    [string]$Family = 'cube',
    [int]$Steps = 300
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1" | Out-Null
$spec = (Get-Content "$Root\protocol.json" -Raw | ConvertFrom-Json).families.$Family
$prep = "$Root\prep\$Family"
$out = "$Root\smoke_$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Force $out | Out-Null
$cache = "$prep\cache\$($spec.dataset)"
function Stage($name, $script, $argv) {
    Write-Host "=== $name"
    & "$PSScriptRoot\run_stage.ps1" -Root $Root -Script "source\$script" -Name "smoke_$name" -ScriptArgs $argv
    if ($LASTEXITCODE -ne 0) { throw "stage $name failed (exit $LASTEXITCODE); see $env:EVENT_WM_RUNS\logs" }
}
Stage 'wm' 'u_wm.py' @('--events', "$prep\events", '--out', "$out\wm", '--wm-steps', $Steps, '--h-steps', $Steps, '--h-width', 2048,
    '--h-goals', 'walk', '--h-walk-pool', 20000, '--h-walk-max', 30, '--protos', 8, '--h-absdiff', '--event-pos-only', '--rest-support')
Stage 'skill' 'u_skill.py' @('--cache', $cache, '--events', "$prep\events", '--train-frames', 0, '--val-frames', 0, '--steps', $Steps,
    '--lr', 0.0003, '--hist-gap', 2, '--chunk', 8, '--release', 10, '--cond', 'full', '--out', "$out\skill")
Stage 'support' 'train_event_support.py' @('--model', "$out\wm\u_model.pt", '--events', "$prep\events", '--out', "$out\support",
    '--steps', $Steps, '--width', 256, '--lr', 0.0003, '--quantile', 0.02, '--seed', 61006)
Stage 'loop' 'u_closed_loop.py' @('--env', $spec.env, '--model', "$out\support\u_model.pt", '--skill', "$out\skill\u_skill.pt",
    '--state-layout', "$prep\front\layout.json", '--events', "$prep\events", '--cache', $cache, '--tasks', 1, '--episodes', 1,
    '--workers', 1, '--max-expansions', 2000, '--seed', 6, '--out', "$out\loop")
Write-Host "SMOKE_OK $out"
