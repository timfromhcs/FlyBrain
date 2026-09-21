# FlyBrain System Doctor & Preflight Diagnostic Tool (V8/V9)
# Requires PowerShell 5.1+ or PowerShell Core 7+ on Windows

param (
    [switch]$Json = $false
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
    Write-Host " FlyBrain V8/V9 Autonomous System Doctor  " -ForegroundColor Cyan
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
