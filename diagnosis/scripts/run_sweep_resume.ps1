# Self-healing closed-loop sweep: scripts/18 resumes from its output CSV, so
# when the python dies natively (MuJoCo/driver on Windows, no traceback) we
# just relaunch it until every (task, seed) pair is complete or the attempt
# budget runs out. Usage:
#   .\scripts\run_sweep_resume.ps1            # default 3 tasks x 16 episodes
param(
    [int]$ExpectedRows = 96,
    [int]$MaxAttempts = 12,
    [string]$OutCsv = "results\metaworld_closed_loop.csv"
)

$ErrorActionPreference = 'Continue'
for ($i = 1; $i -le $MaxAttempts; $i++) {
    "ATTEMPT ${i}/${MaxAttempts} $(Get-Date -Format HH:mm:ss)"
    .\scripts\run_with_watchdog.ps1 -LogName "cl_sweep_r$i" -PyArgs `
        'scripts/18_closed_loop_eval.py', `
        '--config', 'configs/diagnostic_metaworld.yaml', `
        '--model', 'dino_wm_metaworld', `
        '--probe', 'checkpoints/object_probe_dino_wm_metaworld.pt', `
        '--dyn-head', 'checkpoints/object_dynamics_dino_wm_metaworld.pt', `
        '--tasks', 'mw-reach', 'mw-push', 'mw-pick-place', `
        '--episodes', '16', '--out', $OutCsv
    $rows = 0
    if (Test-Path $OutCsv) { $rows = (Import-Csv $OutCsv).Count }
    "ATTEMPT ${i} ended with $rows/$ExpectedRows rows"
    if ($rows -ge $ExpectedRows) { "SWEEP COMPLETE ($rows rows)"; break }
    Start-Sleep -Seconds 15   # let the driver settle before relaunch
}
