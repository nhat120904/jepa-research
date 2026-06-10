# Run a diagnosis script under a RAM watchdog so a memory blow-up aborts the
# process instead of freezing the machine (16 GB box; Windows sysmem fallback
# turns CUDA OOM into system-wide thrash). Usage:
#   .\scripts\run_with_watchdog.ps1 -PyArgs 'scripts/12_boundary_diagnostic.py','--config','configs/diagnostic_metaworld.yaml' -LogName bb_metaworld
param(
    [Parameter(Mandatory = $true)][string[]]$PyArgs,
    [Parameter(Mandatory = $true)][string]$LogName,
    [double]$MinFreeGB = 1.2
)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:CAI_JEPA_BB_PREDICT_ROWS) { $env:CAI_JEPA_BB_PREDICT_ROWS = '256' }
if (-not $env:CAI_JEPA_TORCH_THREADS) { $env:CAI_JEPA_TORCH_THREADS = '2' }

$outLog = "results\logs\$LogName.out.log"
$errLog = "results\logs\$LogName.err.log"
New-Item -ItemType Directory -Force results\logs | Out-Null

$start = Get-Date
$p = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList $PyArgs `
    -NoNewWindow -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
try { $p.PriorityClass = 'BelowNormal' } catch {}

$peakGpuMiB = 0
$peakWsGB = 0.0
$killed = $false
while (-not $p.HasExited) {
    Start-Sleep -Seconds 10
    try {
        $g = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($g -gt $peakGpuMiB) { $peakGpuMiB = $g }
    } catch {}
    try {
        $p.Refresh()
        $ws = $p.WorkingSet64 / 1GB
        if ($ws -gt $peakWsGB) { $peakWsGB = $ws }
    } catch {}
    $freeGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($freeGB -lt $MinFreeGB) {
        "WATCHDOG: free RAM $([math]::Round($freeGB,2)) GB < $MinFreeGB GB -> killing PID $($p.Id)"
        Stop-Process -Id $p.Id -Force -Confirm:$false
        $killed = $true
        break
    }
}
$p.WaitForExit()
$wall = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
"EXIT=$($p.ExitCode) KILLED=$killed WALL_MIN=$wall PEAK_GPU_MIB=$peakGpuMiB PEAK_WS_GB=$([math]::Round($peakWsGB,1))"
