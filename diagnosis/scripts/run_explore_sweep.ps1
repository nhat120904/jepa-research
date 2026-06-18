# Self-healing grounded-exploration sweep (option B): l2 vs hdyn vs hexp on the
# contact tasks. scripts/18 resumes from its CSV, so we relaunch under the RAM
# watchdog until every (task, seed) pair is complete or the attempt budget runs
# out (the python dies natively now and then: MuJoCo/driver on Windows).
#   .\scripts\run_explore_sweep.ps1
param(
    [int]$Episodes = 16,
    [string[]]$Tasks = @('mw-push', 'mw-pick-place'),
    [string[]]$Arms = @('l2', 'hexp'),
    [double]$LambdaApp = 1.0,
    [double]$LambdaObj = 1.0,
    [int]$MaxAttempts = 16,
    [string]$OutCsv = "results\metaworld_grounded_explore.csv"
)

$ExpectedRows = $Episodes * $Tasks.Count * $Arms.Count
$ErrorActionPreference = 'Continue'
for ($i = 1; $i -le $MaxAttempts; $i++) {
    "ATTEMPT ${i}/${MaxAttempts} $(Get-Date -Format HH:mm:ss)  (lambda_app=$LambdaApp lambda_obj=$LambdaObj)"
    $pyArgs = @(
        'scripts/18_closed_loop_eval.py',
        '--config', 'configs/diagnostic_metaworld.yaml',
        '--model', 'dino_wm_metaworld',
        '--probe', 'checkpoints/object_probe_dino_wm_metaworld.pt',
        '--dyn-head', 'checkpoints/object_dynamics_dino_wm_metaworld.pt',
        '--ee-probe', 'checkpoints/ee_probe_dino_wm_metaworld.pt',
        '--lambda-app', "$LambdaApp", '--lambda-obj', "$LambdaObj",
        '--tasks') + $Tasks + @('--arms') + $Arms + @(
        '--episodes', "$Episodes", '--out', $OutCsv)
    .\scripts\run_with_watchdog.ps1 -LogName "explore_r$i" -PyArgs $pyArgs
    $rows = 0
    if (Test-Path $OutCsv) { $rows = (Import-Csv $OutCsv).Count }
    "ATTEMPT ${i} ended with $rows/$ExpectedRows rows"
    if ($rows -ge $ExpectedRows) { "SWEEP COMPLETE ($rows rows)"; break }
    Start-Sleep -Seconds 15
}
