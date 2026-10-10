# Method pipeline for one family on this PC (method/README.md). Stages run in order through local/run_stage.ps1 (RAM
# watchdog, one GPU job at a time); finished stages are skipped. Same arguments for every family.
#   .\method\run_family.ps1 -Family puzzle -Stages objects,events,wm,relabel,wm2,support,h
# Object track (default): token entities (SeeThrough codes) -> objects -> events -> world model (round 1) -> relabel by
# explanation -> world model (round 2) -> support -> cost-to-go -> skill -> closed loop (loop_scripted: PRIVILEGED press
# arm, puzzle only, a diagnostic of the high level).
# The token track (events.py on token entities; outputs events\, model\) is kept as history: -Stages tok_events,tok_wm,...
param(
    [Parameter(Mandatory = $true)][string]$Family,   # cube / puzzle / scene (dev), or any name with -Run -DataEnv -EvalEnv (held-out)
    [string]$Run = '',      # held-out: front-end run dir (local/run_sm2.ps1 output)
    [string]$DataEnv = '',  # held-out: play dataset, e.g. visual-puzzle-3x3-play-v0
    [string]$EvalEnv = '',  # held-out: evaluation env, e.g. visual-puzzle-3x3-v0
    [string[]]$Stages = @('entities', 'objects', 'events', 'wm', 'relabel', 'wm2', 'support', 'h', 'skill', 'loop'),
    [int]$LoopSeed = 0,
    [int]$LoopEpisodes = 6
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$root = 'E:\jepa-data\event_wm\unified_local'
$dev = @{
    cube   = @('cube_grow_20261008093456', 'visual-cube-triple-play-v0', 'visual-cube-triple-v0');
    puzzle = @('puzzle_grow_20261008095020', 'visual-puzzle-4x5-play-v0', 'visual-puzzle-4x5-v0');
    scene  = @('scene_grow_20261008091629', 'visual-scene-play-v0', 'visual-scene-v0')
}
if ($dev.ContainsKey($Family)) {
    $fam = $dev[$Family]; $run = "E:\jepa-data\event_wm\scene_memory_v2\$($fam[0])"
} else {
    if (-not ($Run -and $DataEnv -and $EvalEnv)) { throw "held-out family $Family needs -Run, -DataEnv and -EvalEnv" }
    $fam = @((Split-Path $Run -Leaf), $DataEnv, $EvalEnv); $run = $Run
}
$cache = "E:\jepa-data\event_wm\cache\$($fam[1])"
$m = "$run\method"
$failed = $false
function Stage($name, $script, $argv, $done) {
    if (Test-Path $done) { "=== $Family $name done, skipped"; return }
    "=== $Family $name $(Get-Date -Format s)"
    & "$repo\local\run_stage.ps1" -Root $root -Script "$repo\method\$script" -Name "method_${Family}_$name" -ScriptArgs $argv
    if (-not (Test-Path $done)) { "=== $Family $name FAILED (no $done): later stages not run"; $script:failed = $true }
}
foreach ($s in $Stages) {
    if ($failed) { break }
    switch ($s) {
        'entities' { Stage 'entities' 'memory_entities.py' @('--run', "$run\train", '--cache', $cache, '--see-cache', "$run\events", '--out', "$m\entities") "$m\entities\entities_report.json" }
        'objects'  { Stage 'objects' 'objects.py' @('--run', "$run\train", '--cache', $cache, '--tokens', "$m\entities", '--out', "$m\objects") "$m\objects\discover.json" }
        'events'   { Stage 'events' 'events_objects.py' @('--entities', "$m\objects", '--cache', $cache, '--per-frame', '--out', "$m\obj_events") "$m\obj_events\report.json" }
        # round 1 world model -> relabel by explanation (component 8b) -> round 2 world model, used from here on
        'wm'       { Stage 'wm' 'world_model.py' @('--stage', 'wm', '--events', "$m\obj_events", '--out', "$m\obj_model") "$m\obj_model\wm_stage.pt" }
        'relabel'  { Stage 'relabel' 'relabel_events.py' @('--events', "$m\obj_events", '--model', "$m\obj_model\wm_stage.pt", '--out', "$m\obj_events_r2") "$m\obj_events_r2\relabel_report.json" }
        'wm2'      { Stage 'wm2' 'world_model.py' @('--stage', 'wm', '--events', "$m\obj_events_r2", '--out', "$m\obj_model_r2") "$m\obj_model_r2\wm_stage.pt" }
        'support'  { Stage 'support' 'train_support.py' @('--model', "$m\obj_model_r2\wm_stage.pt", '--events', "$m\obj_events_r2", '--out', "$m\obj_model_r2") "$m\obj_model_r2\model_support.pt" }
        'h'        { Stage 'h' 'world_model.py' @('--stage', 'h', '--events', "$m\obj_events_r2", '--init', "$m\obj_model_r2\model_support.pt", '--out', "$m\obj_model_r2") "$m\obj_model_r2\model.pt" }
        'skill'    { Stage 'skill' 'skill.py' @('--cache', $cache, '--events', "$m\obj_events_r2", '--out', "$m\obj_skill") "$m\obj_skill\skill.pt" }
        'loop'     { Stage "loop_seed$LoopSeed" 'closed_loop_objects.py' @('--env', $fam[2], '--model', "$m\obj_model_r2\model.pt", '--skill', "$m\obj_skill\skill_best.pt",
                        '--front', "$run\train", '--objects', "$m\objects", '--tokens', "$m\entities", '--events', "$m\obj_events_r2", '--cache', $cache,
                        '--episodes', "$LoopEpisodes", '--seed', "$LoopSeed", '--out', "$m\obj_loop_seed$LoopSeed") "$m\obj_loop_seed$LoopSeed\closed_loop.json" }
        'loop_scripted' { Stage "loop_scripted_seed$LoopSeed" 'closed_loop_objects.py' @('--env', $fam[2], '--model', "$m\obj_model_r2\model.pt", '--skill', "$m\obj_skill\skill_best.pt",
                        '--front', "$run\train", '--objects', "$m\objects", '--tokens', "$m\entities", '--events', "$m\obj_events_r2", '--cache', $cache, '--low', 'scripted',
                        '--episodes', "$LoopEpisodes", '--seed', "$LoopSeed", '--out', "$m\obj_loop_scripted_seed$LoopSeed") "$m\obj_loop_scripted_seed$LoopSeed\closed_loop.json" }
        # token track (history)
        'tok_events'  { Stage 'tok_events' 'events.py' @('--entities', "$m\entities", '--cache', $cache, '--out', "$m\events") "$m\events\report.json" }
        'tok_wm'      { Stage 'tok_wm' 'world_model.py' @('--stage', 'wm', '--events', "$m\events", '--out', "$m\model") "$m\model\wm_stage.pt" }
        'tok_support' { Stage 'tok_support' 'train_support.py' @('--model', "$m\model\wm_stage.pt", '--events', "$m\events", '--out', "$m\model") "$m\model\model_support.pt" }
        'tok_h'       { Stage 'tok_h' 'world_model.py' @('--stage', 'h', '--events', "$m\events", '--init', "$m\model\model_support.pt", '--out', "$m\model") "$m\model\model.pt" }
        'tok_skill'   { Stage 'tok_skill' 'skill.py' @('--cache', $cache, '--events', "$m\events", '--out', "$m\skill") "$m\skill\skill.pt" }
        'tok_loop'    { Stage "tok_loop_seed$LoopSeed" 'closed_loop.py' @('--env', $fam[2], '--model', "$m\model\model.pt", '--skill', "$m\skill\skill_best.pt", '--front', "$run\train",
                        '--entities', "$m\entities", '--events', "$m\events", '--cache', $cache, '--episodes', "$LoopEpisodes", '--seed', "$LoopSeed", '--out', "$m\loop_seed$LoopSeed") "$m\loop_seed$LoopSeed\closed_loop.json" }
    }
}
"FAMILY_DONE $Family"
