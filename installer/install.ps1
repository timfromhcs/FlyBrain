# FlyBrain Autonomous One-Line Installer & Self-Healing Setup Script (V10)
# Usage:
#   powershell -ExecutionPolicy Bypass -File installer/install.ps1
# Or with repair:
#   powershell -ExecutionPolicy Bypass -File installer/install.ps1 -Repair

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

# 1. Resolve Canonical Version
$version = "10.0.0"
$versionPy = Join-Path $PSScriptRoot "..\src\version.py"
if (Test-Path $versionPy) {
    $content = Get-Content $versionPy -Raw
    if ($content -match 'VERSION\s*=\s*"([^"]+)"') {
        $version = $matches[1]
    }
}
Write-Host "Installing Version: $version" -ForegroundColor Green

# 2. Self-Healing Directory Layout (Section 73)
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

# 3. Run Preflight Doctor Check
$doctorScript = Join-Path $PSScriptRoot "doctor.ps1"
if ((-not $SkipDoctor) -and (Test-Path $doctorScript)) {
    Write-Host "`nRunning Preflight System Doctor..." -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File $doctorScript
}

# 4. Write/Repair User Runtime Configuration
$configFile = Join-Path $TargetDir "config.json"
$needsConfig = $Repair -or (-not (Test-Path $configFile))
if (-not $needsConfig) {
    try {
        $testJson = Get-Content $configFile -Raw | ConvertFrom-Json
        if (-not $testJson.version) { $needsConfig = $true }
    } catch {
        $needsConfig = $true
    }
}

if ($needsConfig) {
    $config = @{
        version = $version
        installed_at = (Get-Date).ToString("o")
        root_directory = $TargetDir
        profile = "BALANCED"
        storage_provider = "LocalFilesystem"
        vulkan_enabled = (Test-Path (Join-Path $env:SystemRoot "System32\vulkan-1.dll"))
    }
    $config | ConvertTo-Json -Depth 4 | Set-Content -Path $configFile -Encoding utf8
    Write-Host "  [+] Runtime configuration written to $configFile" -ForegroundColor Green
}

# 5. Create/Repair Launcher Shortcut
$launchBat = Join-Path $TargetDir "launch.bat"
if ($Repair -or (-not (Test-Path $launchBat))) {
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
}

# 6. Emit Installation Manifest
$installManifest = Join-Path $TargetDir "diagnostics\installation_manifest.json"
$diagDir = Join-Path $TargetDir "diagnostics"
if (-not (Test-Path $diagDir)) { New-Item -ItemType Directory -Path $diagDir -Force | Out-Null }

$manifestData = @{
    version = $version
    installed_at = (Get-Date).ToString("o")
    status = "INSTALLED"
    repair_mode = [bool]$Repair
    target_dir = $TargetDir
    components = @{
        config_json = (Test-Path $configFile)
        launch_bat = (Test-Path $launchBat)
        subdirs_count = $subdirs.Length
    }
}
$manifestData | ConvertTo-Json -Depth 4 | Set-Content -Path $installManifest -Encoding utf8

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " FlyBrain Installation Complete & Ready for Execution     " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
