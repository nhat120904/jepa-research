# Object-track queue on this PC (method/README.md), one stage at a time: the dev families, then the held-out environments
# (front end with the same hyper-parameters via run_sm2.ps1, then the same object chain; no code or setting changes).
#   .\local\run_object_queue.ps1                      # everything; finished stages are skipped
#   .\local\run_object_queue.ps1 -Dev puzzle -HeldOut @()
param(
    [string[]]$Dev = @('puzzle', 'cube', 'scene'),
    [string[]]$HeldOut = @('cube_single', 'puzzle3x3', 'puzzle4x4'),
    [int]$LoopEpisodes = 6
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$rf = "$repo\method\run_family.ps1"
$chain = @('entities', 'objects', 'events', 'wm', 'relabel', 'wm2', 'support', 'h', 'skill', 'loop')
foreach ($f in $Dev) {
    "##### DEV $f $(Get-Date -Format s)"
    & $rf -Family $f -Stages $chain -LoopEpisodes $LoopEpisodes
}
$held = @{
    cube_single = @('cube', 'visual-cube-single-play-v0', 'visual-cube-single-v0');
    puzzle3x3   = @('puzzle', 'visual-puzzle-3x3-play-v0', 'visual-puzzle-3x3-v0');
    puzzle4x4   = @('puzzle', 'visual-puzzle-4x4-play-v0', 'visual-puzzle-4x4-v0')
}
foreach ($h in $HeldOut) {
    $cfg = $held[$h]
    $out = "E:\jepa-data\event_wm\scene_memory_v2\heldout_$h"
    "##### HELD-OUT $h front end $(Get-Date -Format s)"
    try { & "$repo\local\run_sm2.ps1" -Family $cfg[0] -DataEnv $cfg[1] -Out $out } catch { "front end failed: $_"; continue }
    "##### HELD-OUT $h chain $(Get-Date -Format s)"
    & $rf -Family $h -Run $out -DataEnv $cfg[1] -EvalEnv $cfg[2] -Stages $chain -LoopEpisodes $LoopEpisodes
}
"QUEUE_DONE $(Get-Date -Format s)"
