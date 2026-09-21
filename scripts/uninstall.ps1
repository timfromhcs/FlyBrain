# FlyBrain Uninstaller Script
param (
    [string]$TargetDir = "$env:LOCALAPPDATA\FlyBrain",
    [switch]$KeepData = $true
)

Write-Host "FlyBrain Uninstaller" -ForegroundColor Yellow
if (Test-Path $TargetDir) {
    if ($KeepData) {
        Write-Host "Preserving user data (models, world, backups, data)..." -ForegroundColor Cyan
        Remove-Item (Join-Path $TargetDir "app") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $TargetDir "runtime") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $TargetDir "launch.bat") -Force -ErrorAction SilentlyContinue
        Write-Host "Application binaries removed from $TargetDir. User data preserved." -ForegroundColor Green
    } else {
        Remove-Item $TargetDir -Recurse -Force
        Write-Host "Entire FlyBrain directory $TargetDir removed." -ForegroundColor Green
    }
} else {
    Write-Host "FlyBrain is not installed at $TargetDir." -ForegroundColor Yellow
}
