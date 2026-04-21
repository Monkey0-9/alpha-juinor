@echo off
REM Alpha Junior - 24/7 Production Runner
REM Auto-restarts on crash, logs everything

echo ==========================================
echo   Alpha Junior - 24/7 Production Mode
echo ==========================================
echo.
echo This window must stay open.
echo The server will auto-restart if it crashes.
echo.
echo Logs: logs\alpha_junior.log
echo.

if not exist logs mkdir logs

cd /d "%~dp0"

:loop
echo [%date% %time%] Starting Alpha Junior...
echo [%date% %time%] Starting... >> logs\alpha_junior.log

python runner.py >> logs\alpha_junior.log 2>&1

echo.
echo [%date% %time%] Server crashed or stopped. Restarting in 10 seconds...
echo [%date% %time%] Restarting... >> logs\alpha_junior.log
echo.
timeout /t 10 /nobreak >nul
goto loop
