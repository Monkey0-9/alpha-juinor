@echo off
REM HugeFunds - One-Click Windows Launcher
REM Runs backend + live monitor in separate windows

cd /d "%~dp0"

echo ========================================
echo  HUGE FUNDS - Starting 24/7 Trading
echo ========================================
echo.

REM Kill any existing uvicorn processes
taskkill /F /FI "COMMANDLINE eq *uvicorn*" 2>nul
taskkill /F /FI "COMMANDLINE eq *live_monitor*" 2>nul
timeout /t 2 /nobreak >nul

REM Start Backend Server in new window
start "HUGEFUNDS BACKEND" cmd /k "cd backend && ..\.venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info"

echo [OK] Backend starting on port 8000...
timeout /t 5 /nobreak >nul

REM Start Live Monitor in new window
start "HUGEFUNDS MONITOR" cmd /k "..\.venv\Scripts\python live_monitor.py"

echo [OK] Live monitor starting...
echo.
echo ========================================
echo  ALL SYSTEMS RUNNING
echo ========================================
echo.
echo Access Points:
echo   http://localhost:8000         - Dashboard
echo   http://localhost:8000/docs      - API Docs
echo.
echo Trade Commands:
echo   Buy:  curl -X POST "http://localhost:8000/api/alpaca/buy?symbol=AAPL^&qty=10"
echo   Sell: curl -X POST "http://localhost:8000/api/alpaca/sell?symbol=AAPL^&qty=10"
echo.
echo Press any key to close this window (backend keeps running)
pause >nul
