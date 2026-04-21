@echo off
REM Alpha Junior - Simple Local Launcher

echo ==========================================
echo    Alpha Junior - Starting
echo ==========================================
echo.

echo [1/2] Installing dependencies...
python -m pip install -q flask flask-cors
if errorlevel 1 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo [2/2] Starting server...
echo.
echo ==========================================
echo    OPEN http://localhost:5000 in your browser
echo ==========================================
echo.
echo Press Ctrl+C to stop
echo.

python app.py

pause
