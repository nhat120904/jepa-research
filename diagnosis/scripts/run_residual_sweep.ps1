# Self-healing predictor-fix sweep (option C): frozen predictor (l2) vs corrected
# predictor F+Δ (l2c), plain L2 cost both, on the contact tasks. Holds the cost
# at L2 so the predictor is the only variable. scripts/18 resumes from its CSV.
#   .\scripts\run_residual_sweep.ps1                 # 8 ep first signal
#   .\scripts\run_residual_sweep.ps1 -Episodes 16    # full
param(
    [int]$Episodes = 8,
    [string[]]$Tasks = @('mw-push', 'mw-pick-place'),
    [string[]]$Arms = @('l2', 'l2c'),
    [int]$MaxAttempts = 20,
    [string]$OutCsv = "results\metaworld_residual.csv"
)

$ExpectedRows = $Episodes * $Tasks.Count * $Arms.Count
$ErrorActionPreference = 'Continue'
for ($i = 1; $i -le $MaxAttempts; $i++) {
    "ATTEMPT ${i}/${MaxAttempts} $(Get-Date -Format HH:mm:ss)"
    $pyArgs = @(
        'scripts/18_closed_loop_eval.py',
        '--config', 'configs/diagnostic_metaworld.yaml',
        '--model', 'dino_wm_metaworld',
        '--probe', 'checkpoints/object_probe_dino_wm_metaworld.pt',
        '--dyn-head', 'checkpoints/object_dynamics_dino_wm_metaworld.pt',
        '--residual-head', 'checkpoints/residual_predictor_dino_wm_metaworld.pt',
        '--tasks') + $Tasks + @('--arms') + $Arms + @(
        '--episodes', "$Episodes", '--out', $OutCsv)
    .\scripts\run_with_watchdog.ps1 -LogName "residual_r$i" -PyArgs $pyArgs
    $rows = 0
    if (Test-Path $OutCsv) { $rows = (Import-Csv $OutCsv).Count }
    "ATTEMPT ${i} ended with $rows/$ExpectedRows rows"
    if ($rows -ge $ExpectedRows) { "SWEEP COMPLETE ($rows rows)"; break }
    Start-Sleep -Seconds 15
}
