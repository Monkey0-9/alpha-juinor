@echo off
REM ============================================================================
REM NEXUS INSTITUTIONAL - LIVE PAPER TRADING SETUP
REM ============================================================================
REM One-time setup for live paper trading monitoring
REM Run this once to install dependencies
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║              NEXUS INSTITUTIONAL - LIVE TRADING SETUP                     ║
echo ║                    Installing Dependencies...                             ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo    Please install Python 3.9+ from https://www.python.org
    echo    Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo ✓ Python found
python --version

echo.
echo Installing required packages...
echo.

REM Install core dependencies
pip install --upgrade pip
pip install feedparser requests textblob
pip install numpy pandas
pip install asyncio

echo.
echo ✓ Installation complete!
echo.
echo Next steps:
echo   1. Run: start_live_monitor.bat
echo   2. View live dashboard: http://localhost:8000/live_dashboard.html
echo   3. Press Ctrl+C to stop monitoring
echo.
pause
