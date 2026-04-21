@echo off
REM Alpha Junior - Simple 24/7 Runner

echo ==========================================
echo   Alpha Junior - 24/7 Mode
echo ==========================================
echo.
echo Starting server... Keep this window open!
echo.
echo Access: http://localhost:5000
echo.

cd /d "%~dp0"

:loop
python runner.py
if %errorlevel% neq 0 (
    echo.
    echo Server stopped. Restarting in 5 seconds...
    timeout /t 5 /nobreak >nul
    goto loop
)

echo Server exited normally.
pause
