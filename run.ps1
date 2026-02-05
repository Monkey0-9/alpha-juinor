# run.ps1
$ErrorActionPreference = "Stop"

Write-Host "Starting Mini Quant Fund..." -ForegroundColor Cyan

# Define venv path
$VenvPath = "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Check if venv exists
if (-not (Test-Path $VenvPath)) {
    Write-Error "Virtual environment not found at $VenvPath. Please create it first."
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Gray
. $VenvPath

# Ensure dependencies are installed (fast check)
# python -m pip install -r requirements.txt | Out-Null

# Run main.py
Write-Host "Running main.py..." -ForegroundColor Green
python main.py
