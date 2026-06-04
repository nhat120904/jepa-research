param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [double]$FreeGbFloor = 3.0,
    [double]$PrivateGbCeiling = 10.0,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"

$DiagnosisRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $DiagnosisRoot

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*#" -or $_ -notmatch "=") { return }
        $parts = $_ -split "=", 2
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HDF5_USE_FILE_LOCKING = "FALSE"
$env:CAI_JEPA_TORCH_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:NUMEXPR_NUM_THREADS = "2"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path (Get-Location) "logs\05_${Tag}_${stamp}.out.log"
$stderr = Join-Path (Get-Location) "logs\05_${Tag}_${stamp}.err.log"
$monitor = Join-Path (Get-Location) "logs\05_${Tag}_${stamp}.monitor.log"

$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$args = @("scripts\05_run_diagnostic.py", "--config", $Config)
$proc = Start-Process -FilePath $python -ArgumentList $args -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden

try { $proc.PriorityClass = "BelowNormal" } catch {}

function Get-DescendantProcessIds([int]$RootPid) {
    $seen = @{}
    $frontier = @($RootPid)
    while ($frontier.Count -gt 0) {
        $current = $frontier[0]
        if ($frontier.Count -eq 1) {
            $frontier = @()
        } else {
            $frontier = $frontier[1..($frontier.Count - 1)]
        }
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$current" -ErrorAction SilentlyContinue
        foreach ($child in $children) {
            if (-not $seen.ContainsKey($child.ProcessId)) {
                $seen[$child.ProcessId] = $true
                $frontier += [int]$child.ProcessId
            }
        }
    }
    return @($seen.Keys)
}

"started pid=$($proc.Id) config=$Config stdout=$stdout stderr=$stderr free_floor_gb=$FreeGbFloor private_ceiling_gb=$PrivateGbCeiling" |
    Out-File -FilePath $monitor -Encoding utf8

while ($true) {
    $proc.Refresh()
    $childIds = @(Get-DescendantProcessIds $proc.Id)
    $ids = @($proc.Id) + $childIds
    $live = @(Get-Process -Id $ids -ErrorAction SilentlyContinue)

    foreach ($p in $live) {
        try { $p.PriorityClass = "BelowNormal" } catch {}
    }

    $privateGb = 0.0
    if ($live.Count -gt 0) {
        $privateGb = (($live | Measure-Object -Property PrivateMemorySize64 -Sum).Sum / 1GB)
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGb = $os.FreePhysicalMemory / 1MB
    $cpu = 0.0
    if ($live.Count -gt 0) {
        $cpu = ($live | Measure-Object -Property CPU -Sum).Sum
    }
    $line = "{0:o} pids={1} private_gb={2:n2} free_gb={3:n2} cpu_s={4:n1}" -f `
        (Get-Date), (($live | ForEach-Object { $_.Id }) -join ","), $privateGb, $freeGb, $cpu
    $line | Out-File -FilePath $monitor -Append -Encoding utf8

    if ($freeGb -lt $FreeGbFloor -or $privateGb -gt $PrivateGbCeiling) {
        "guard_stop reason=memory free_gb={0:n2} private_gb={1:n2}" -f $freeGb, $privateGb |
            Out-File -FilePath $monitor -Append -Encoding utf8
        foreach ($p in $live) {
            try { Stop-Process -Id $p.Id -Force } catch {}
        }
        exit 90
    }

    if ($proc.HasExited) {
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        "exited code=$exitCode" | Out-File -FilePath $monitor -Append -Encoding utf8
        exit $exitCode
    }

    Start-Sleep -Seconds $PollSeconds
}
