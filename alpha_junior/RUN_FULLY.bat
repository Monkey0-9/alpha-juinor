@echo off
REM Alpha Junior - FULLY AUTOMATED SYSTEM
REM Kernel-level operation with full visibility

echo.
echo ============================================
echo   Alpha Junior - FULL SYSTEM LAUNCHER
echo ============================================
echo.
echo [1] Checking system requirements...
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found! Install Python 3.11+ from python.org
    pause
    exit /b 1
)
echo [OK] Python found

REM Check dependencies
echo [2] Installing dependencies...
pip install -q flask flask-cors requests 2>nul
echo [OK] Dependencies ready

REM Check .env
echo [3] Checking configuration...
if not exist .env (
    echo [X] .env file not found!
    pause
    exit /b 1
)

findstr /C:"ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE" .env >nul
if not errorlevel 1 (
    echo [!] WARNING: You need to add your Alpaca API keys!
    echo [!] Edit .env file and replace YOUR_SECRET_KEY_HERE
    echo [!] Get keys from: https://alpaca.markets/
    echo.
    choice /C YN /M "Continue anyway (Y/N)"
    if errorlevel 2 exit /b 1
)
echo [OK] Configuration checked

echo.
echo ============================================
echo   [4] STARTING ALPHA JUNIOR
echo ============================================
echo.
echo Opening TWO windows:
echo   - Window 1: Server (keep this running)
echo   - Window 2: Dashboard (watch trades live)
echo.
echo Press any key to start...
pause >nul

REM Start server in new window
echo [5] Starting server...
start "Alpha Junior - SERVER (Keep Open)" cmd /k "cd /d %~dp0 ^&^& echo Starting server... ^&^& python runner.py ^&^& echo. ^&^& echo Server stopped. Press any key to close. ^&^& pause"

REM Wait for server to start
timeout /t 5 /nobreak >nul

REM Start dashboard in new window
echo [6] Starting dashboard...
start "Alpha Junior - DASHBOARD" cmd /k "cd /d %~dp0 ^&^& echo Starting dashboard... ^&^& python monitor_dashboard.py ^&^& echo. ^&^& echo Dashboard closed. ^&^& pause"

REM Open browser
echo [7] Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:5000

echo.
echo ============================================
echo   [OK] SYSTEM FULLY RUNNING!
echo ============================================
echo.
echo What is happening now:
echo   1. Server window: Running the trading engine
echo   2. Dashboard window: Showing live trades and P/L
echo   3. Browser: Web interface at http://localhost:5000
echo.
echo What Alpha Junior does:
echo   - Monitors stocks every 5 minutes
echo   - Buys when momentum is strong
echo   - Sells when overbought
"echo   - Target: 50-60%% annual returns"
echo.
echo To stop:
echo   - Close the SERVER window (Ctrl+C)
echo   - Dashboard will stop automatically
echo.
pause
