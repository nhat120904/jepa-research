# Run one stage script on this PC under a RAM watchdog (local stand-in for `sbatch`).
#   .\local\run_stage.ps1 -Root E:\jepa-data\event_wm\unified_local -Script run_prepare.py -Name prep -ScriptArgs '--root','E:\jepa-data\event_wm\unified_local'
#
# -Root   run root made by make_run_root.py: PYTHONPATH = <Root>\source;<Root> (frozen code wins over scripts/).
# -Script path relative to -Root, or absolute.
# Gives every stage its own SLURM_JOB_ID ("local<timestamp>") because the drivers name run directories after it
# and the repo guards require it. One GPU stage at a time on this 12 GB card (see memory: unchunked batches froze the PC).
# The venv's python.exe is a launcher: the real interpreter (and multiprocessing workers) are its descendants, so the
# watchdog sums the working set over the whole tree. Killing the launcher also kills the tree (verified).
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Script,
    [Parameter(Mandatory = $true)][string]$Name,
    [string[]]$ScriptArgs = @(),
    [double]$MinFreeGB = 1.5,
    [switch]$NoWatchdog
)
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\env.ps1" | Out-Null
$env:SLURM_JOB_ID = 'local' + (Get-Date -Format 'yyyyMMddHHmmss')
$env:PYTHONPATH = "$Root\source;$Root"
$scriptPath = if ([IO.Path]::IsPathRooted($Script)) { $Script } else { Join-Path $Root $Script }
$logs = Join-Path $env:EVENT_WM_RUNS 'logs'
$out = Join-Path $logs "$Name`_$($env:SLURM_JOB_ID).out"
$err = Join-Path $logs "$Name`_$($env:SLURM_JOB_ID).err"
# Start-Process joins arguments with spaces and does no quoting of its own.
$argList = @($scriptPath) + $ScriptArgs | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }

function Get-TreeWorkingSetGB([int]$RootPid) {
    $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, WorkingSetSize
    $ids = [System.Collections.Generic.HashSet[int]]::new(); [void]$ids.Add($RootPid)
    do { $n = $ids.Count; foreach ($q in $all) { if ($ids.Contains([int]$q.ParentProcessId)) { [void]$ids.Add([int]$q.ProcessId) } } } while ($ids.Count -ne $n)
    $sum = 0.0; foreach ($q in $all) { if ($ids.Contains([int]$q.ProcessId)) { $sum += $q.WorkingSetSize } }
    return $sum / 1GB
}

$start = Get-Date
$p = Start-Process -FilePath $EventWmPy -ArgumentList $argList -WorkingDirectory $Root -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
$null = $p.Handle   # keep the process handle: without it Windows PowerShell reports ExitCode as $null
try { $p.PriorityClass = 'BelowNormal' } catch {}
"JOB=$($env:SLURM_JOB_ID) PID=$($p.Id) LOG=$out"
$peakGpu = 0; $peakWs = 0.0; $killed = $false
while (-not $p.HasExited) {
    Start-Sleep -Seconds 10
    try { $g = [int](@(nvidia-smi --query-gpu=memory.used '--format=csv,noheader,nounits')[0]); if ($g -gt $peakGpu) { $peakGpu = $g } } catch {}
    try { $ws = Get-TreeWorkingSetGB $p.Id; if ($ws -gt $peakWs) { $peakWs = $ws } } catch {}
    if (-not $NoWatchdog) {
        $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
        if ($free -lt $MinFreeGB) {
            "WATCHDOG: free RAM $([math]::Round($free, 2)) GB < $MinFreeGB GB -> killing PID $($p.Id) and its children"
            & taskkill /PID $p.Id /T /F | Out-Null; $killed = $true; break
        }
    }
}
$p.WaitForExit()
"EXIT=$($p.ExitCode) KILLED=$killed WALL_MIN=$([math]::Round(((Get-Date) - $start).TotalMinutes, 1)) PEAK_GPU_MIB=$peakGpu PEAK_WS_GB=$([math]::Round($peakWs, 1)) JOB=$($env:SLURM_JOB_ID)"
exit $p.ExitCode
