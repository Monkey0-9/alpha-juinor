@echo off
REM Alpha Junior - FULLY LOCAL LAUNCHER
REM No Docker, no PostgreSQL, no Redis needed!

echo ============================================
echo   Alpha Junior - LOCAL MODE (No Docker!)
echo ============================================
echo.

REM Check Python
echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11+ from https://python.org/downloads
    pause
    exit /b 1
)
python --version
echo.

REM Install packages
echo [2/3] Installing packages...
cd backend
python -m pip install -q -r requirements_local.txt
if errorlevel 1 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)
echo [OK] Packages installed
echo.

REM Start server
echo [3/3] Starting server...
echo.
echo ============================================
echo   STARTING ALPHA JUNIOR...
echo ============================================
echo.
echo API:    http://localhost:8000
echo Docs:   http://localhost:8000/api/v1/docs
echo.
echo Press Ctrl+C to stop
echo.

python -m uvicorn app.main_local:app --host 0.0.0.0 --port 8000 --reload

pause
