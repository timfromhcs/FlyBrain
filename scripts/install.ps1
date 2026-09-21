# FlyBrain Autonomous One-Line Installer & Self-Healing Setup Script (V8/V9)
# Usage:
#   irm https://github.com/timfromhcs/FlyBrain/releases/latest/download/install.ps1 | iex
# Or run locally:
#   powershell -ExecutionPolicy Bypass -File scripts/install.ps1

[CmdletBinding()]
param (
    [string]$TargetDir = "$env:LOCALAPPDATA\FlyBrain",
    [switch]$Repair = $false,
    [switch]$SkipDoctor = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       FlyBrain Autonomous Self-Healing Installer         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Target Installation Root: $TargetDir" -ForegroundColor Yellow

# 1. Self-Healing Directory Layout (Section 73)
$subdirs = @(
    "app", "runtime", "models", "data", "world", "assets",
    "cache", "backups", "logs", "diagnostics", "exports"
)

foreach ($sd in $subdirs) {
    $fullPath = Join-Path $TargetDir $sd
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  [+] Created directory: $sd" -ForegroundColor DarkGray
    }
}

# 2. Run Preflight Doctor Check (Section 70)
$doctorScript = Join-Path $PSScriptRoot "doctor.ps1"
if ((-not $SkipDoctor) -and (Test-Path $doctorScript)) {
    Write-Host "`nRunning Preflight System Doctor..." -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File $doctorScript
}

# 3. Write User Runtime Configuration (Section 73)
$configFile = Join-Path $TargetDir "config.json"
$config = @{
    version = "8.0.0"
    installed_at = (Get-Date).ToString("o")
    root_directory = $TargetDir
    profile = "BALANCED"
    storage_provider = "LocalFilesystem"
    vulkan_enabled = (Test-Path (Join-Path $env:SystemRoot "System32\vulkan-1.dll"))
}
$config | ConvertTo-Json -Depth 4 | Set-Content -Path $configFile -Encoding utf8
Write-Host "  [+] Runtime configuration written to $configFile" -ForegroundColor Green

# 4. Create Launcher Shortcut
$launchBat = Join-Path $TargetDir "launch.bat"
$batContent = @"
@echo off
set FLYBRAIN_HOME=$TargetDir
echo Starting FlyBrain Production Workstation...
cd /d "%~dp0"
python src/main.py lab
pause
"@
Set-Content -Path $launchBat -Value $batContent
Write-Host "  [+] Launcher created at $launchBat" -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " FlyBrain Installation Complete & Ready for Execution     " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
