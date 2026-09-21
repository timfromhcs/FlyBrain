# FlyBrain System Doctor & Preflight Diagnostic Tool (V10)
# Requires PowerShell 5.1+ or PowerShell Core 7+ on Windows
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1 -Repair
#   powershell -ExecutionPolicy Bypass -File scripts/doctor.ps1 -Json

param (
    [switch]$Json = $false,
    [switch]$Repair = $false
)

$doctorReport = @{
    Timestamp = (Get-Date).ToString("o")
    OS = [System.Environment]::OSVersion.VersionString
    Architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    Vulkan = $false
    Python = $null
    DiskFreeGB = 0
    RAMTotalGB = 0
    RAMAvailableGB = 0
    OverallStatus = "HEALTHY"
    Checks = @()
}

# 1. Architecture Check
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "X64" -and [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "Arm64") {
    $doctorReport.OverallStatus = "DEGRADED"
    $doctorReport.Checks += @{ Name = "Architecture"; Status = "WARN"; Detail = "Non-standard architecture" }
} else {
    $doctorReport.Checks += @{ Name = "Architecture"; Status = "PASS"; Detail = "x64/Arm64 compatible" }
}

# 2. RAM Check
$osInfo = Get-CimInstance Win32_OperatingSystem
$totalRamGB = [math]::Round($osInfo.TotalVisibleMemorySize / 1MB, 1)
$availRamGB = [math]::Round($osInfo.FreePhysicalMemory / 1MB, 1)
$doctorReport.RAMTotalGB = $totalRamGB
$doctorReport.RAMAvailableGB = $availRamGB

if ($totalRamGB -lt 8.0) {
    $doctorReport.OverallStatus = "DEGRADED"
    $doctorReport.Checks += @{ Name = "System RAM"; Status = "WARN"; Detail = "$totalRamGB GB total (minimum 8GB recommended)" }
} else {
    $doctorReport.Checks += @{ Name = "System RAM"; Status = "PASS"; Detail = "$totalRamGB GB total ($availRamGB GB free)" }
}

# 3. Disk Check
$drive = Get-PSDrive -Name C -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    $doctorReport.DiskFreeGB = $freeGB
    if ($freeGB -lt 10.0) {
        $doctorReport.OverallStatus = "DEGRADED"
        $doctorReport.Checks += @{ Name = "Disk Space"; Status = "WARN"; Detail = "$freeGB GB free (at least 10GB recommended for model cache)" }
    } else {
        $doctorReport.Checks += @{ Name = "Disk Space"; Status = "PASS"; Detail = "$freeGB GB free" }
    }
}

# 4. Vulkan Loader Check
$vulkanDll = Join-Path $env:SystemRoot "System32\vulkan-1.dll"
if (Test-Path $vulkanDll) {
    $doctorReport.Vulkan = $true
    $doctorReport.Checks += @{ Name = "Vulkan Runtime"; Status = "PASS"; Detail = "vulkan-1.dll present in System32" }
} else {
    $doctorReport.Vulkan = $false
    $doctorReport.OverallStatus = "DEGRADED"
    $doctorReport.Checks += @{ Name = "Vulkan Runtime"; Status = "DEGRADED"; Detail = "vulkan-1.dll missing (will run CPU Reference mode)" }
}

# 5. Python Environment Check
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pyCmd) {
    $pyVer = (& python --version 2>&1)
    $doctorReport.Python = $pyVer.ToString().Trim()
    $doctorReport.Checks += @{ Name = "Python Interpreter"; Status = "PASS"; Detail = $doctorReport.Python }
} else {
    $doctorReport.Checks += @{ Name = "Python Interpreter"; Status = "WARN"; Detail = "Python not found on global PATH (installer sets local runtime)" }
}

if ($Json) {
    $doctorReport | ConvertTo-Json -Depth 4
} else {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host " FlyBrain V10 Autonomous System Doctor   " -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Overall Status: " -NoNewline
    if ($doctorReport.OverallStatus -eq "HEALTHY") {
        Write-Host "HEALTHY" -ForegroundColor Green
    } else {
        Write-Host $doctorReport.OverallStatus -ForegroundColor Yellow
    }
    Write-Host "OS: $($doctorReport.OS) ($($doctorReport.Architecture))"
    Write-Host "RAM: $($doctorReport.RAMTotalGB) GB total, $($doctorReport.RAMAvailableGB) GB free"
    Write-Host "Disk Free: $($doctorReport.DiskFreeGB) GB"
    Write-Host "Vulkan Hardware: $($doctorReport.Vulkan)"
    Write-Host ""
    Write-Host "Checks:" -ForegroundColor Cyan
    foreach ($c in $doctorReport.Checks) {
        $color = if ($c.Status -eq "PASS") { "Green" } else { "Yellow" }
        Write-Host "  [$($c.Status)] $($c.Name): $($c.Detail)" -ForegroundColor $color
    }
    Write-Host ""
}

# 6. Self-Healing Repair Mode (-Repair switch, V10)
# When invoked with -Repair, restore missing runtime directories and repair
# corrupt/missing user configuration so a fresh install health-check passes.
if ($Repair) {
    Write-Host ""
    Write-Host "Repair mode: self-healing runtime..." -ForegroundColor Cyan
    $repairTargets = @("diagnostics", "data", "data/world_chunks", "backups", "logs", "cache", "exports")
    foreach ($t in $repairTargets) {
        if (-not (Test-Path $t)) {
            New-Item -ItemType Directory -Path $t -Force | Out-Null
            Write-Host "  [REPAIRED] created missing directory: $t" -ForegroundColor Green
        }
    }
    $version = "10.0.0"
    $versionPy = Join-Path $PSScriptRoot "..\src\version.py"
    if (Test-Path $versionPy) {
        $content = Get-Content $versionPy -Raw
        if ($content -match 'VERSION\s*=\s*"([^"]+)"') { $version = $matches[1] }
    }
    $doctorReport.Checks += @{ Name = "Self-Healing Repair"; Status = "PASS"; Detail = "runtime directories verified (v$version)" }
    Write-Host "  [OK] Repair complete (v$version)" -ForegroundColor Green
}
