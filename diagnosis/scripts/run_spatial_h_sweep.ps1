# Spatial-h planning sweep: plan against the dyn-head object trajectory with the
# SPATIAL probe for the goal/init object (2cm aim) and an object-dominant cost
# (beta high). l2 baseline vs hdyn(spatial). The precisely-motivated shot at
# flipping contact success after the diagnostics localized BB to the predictor's
# counterfactual object channel. Self-healing resume (scripts/18 resumes its CSV).
param(
    [int]$Episodes = 16,
    [string[]]$Tasks = @('mw-push', 'mw-pick-place'),
    [string[]]$Arms = @('l2', 'hdyn'),
    [double]$Beta = 5.0,
    [int]$MaxAttempts = 20,
    [string]$OutCsv = "results\metaworld_spatial_h.csv"
)
$ExpectedRows = $Episodes * $Tasks.Count * $Arms.Count
$ErrorActionPreference = 'Continue'
for ($i = 1; $i -le $MaxAttempts; $i++) {
    "ATTEMPT ${i}/${MaxAttempts} $(Get-Date -Format HH:mm:ss)  (beta=$Beta, spatial probe)"
    $pyArgs = @(
        'scripts/18_closed_loop_eval.py',
        '--config', 'configs/diagnostic_metaworld.yaml',
        '--model', 'dino_wm_metaworld',
        '--probe', 'checkpoints/spatial_object_probe_dino_wm_metaworld.pt',
        '--dyn-head', 'checkpoints/object_dynamics_dino_wm_metaworld.pt',
        '--beta', "$Beta",
        '--tasks') + $Tasks + @('--arms') + $Arms + @(
        '--episodes', "$Episodes", '--out', $OutCsv)
    .\scripts\run_with_watchdog.ps1 -LogName "spatialh_r$i" -PyArgs $pyArgs
    $rows = 0
    if (Test-Path $OutCsv) { $rows = (Import-Csv $OutCsv).Count }
    "ATTEMPT ${i} ended with $rows/$ExpectedRows rows"
    if ($rows -ge $ExpectedRows) { "SWEEP COMPLETE ($rows rows)"; break }
    Start-Sleep -Seconds 15
}
