<#
.SYNOPSIS
    Launches the Mini Quant Fund application using the isolated virtual environment.
.DESCRIPTION
    This script ensures that the application uses the Python interpreter in .venv,
    guaranteeing that all dependencies (statsmodels, etc.) are found.
#>

$VenvPython = "$PSScriptRoot\.venv\Scripts\python.exe"
$MainScript = "$PSScriptRoot\main.py"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython. Please run standard setup."
    exit 1
}

Write-Host "🚀 Launching Mini Quant Fund using isolated environment..." -ForegroundColor Cyan
Write-Host "Python Path: $VenvPython" -ForegroundColor Gray

& $VenvPython $MainScript
