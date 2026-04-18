@echo off
REM ============================================================================
REM NEXUS INSTITUTIONAL v0.3.0 - 24/7 WINDOWS TASK SCHEDULER SETUP
REM ============================================================================
REM This script sets up Windows Task Scheduler to run Nexus 24/7

setlocal enabledelayedexpansion

cd /d C:\mini-quant-fund

echo.
echo =========================================================================
echo  NEXUS INSTITUTIONAL 24/7 - WINDOWS TASK SCHEDULER SETUP
echo =========================================================================
echo.

REM Get the path to python executable
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i

echo Python executable: %PYTHON_EXE%
echo Project directory: %cd%
echo.

REM Create batch file to run in background
echo Creating startup batch file...

(
    @echo off
    setlocal enabledelayedexpansion
    cd /d c:\mini-quant-fund
    %PYTHON_EXE% run_24_7.py --mode backtest --asset-class multi --venues 235
) > c:\mini-quant-fund\run_24_7_startup.bat

echo. 
echo Batch file created: c:\mini-quant-fund\run_24_7_startup.bat
echo.

REM Create task to run on startup
echo Creating Windows Task Scheduler task...
echo.

schtasks /create ^
  /tn "NexusInstitutional24x7" ^
  /tr "c:\mini-quant-fund\run_24_7_startup.bat" ^
  /sc onstart ^
  /ru SYSTEM ^
  /f ^
  /rl highest

if %errorlevel% equ 0 (
    echo [SUCCESS] Task created successfully!
    echo.
    echo Task scheduled as: NexusInstitutional24x7
    echo Run on: System startup
    echo Run as: SYSTEM
    echo Action: Run c:\mini-quant-fund\run_24_7_startup.bat
    echo.
) else (
    echo [ERROR] Failed to create task. Error code: %errorlevel%
    echo.
    echo Troubleshooting:
    echo 1. Run this script as ADMINISTRATOR
    echo 2. Check Windows Event Viewer for errors
    echo 3. Manually create task: 
    echo    schtasks /create /tn "NexusInstitutional24x7" /tr "c:\mini-quant-fund\run_24_7_startup.bat" /sc onstart /ru SYSTEM /rl highest
    echo.
)

REM Show task details
echo Verifying task...
schtasks /query /tn "NexusInstitutional24x7" /v
echo.

REM Show next run time
echo Task Summary:
schtasks /query /tn "NexusInstitutional24x7" /fo list
echo.

echo =========================================================================
echo  SETUP COMPLETE
echo =========================================================================
echo.
echo The Nexus Institutional platform will now run 24/7 on system startup.
echo.
echo To verify it's running:
echo   tasklist | find "python"
echo.
echo To view logs:
echo   type logs\nexus_24_7_*.log
echo.
echo To stop the task:
echo   schtasks /end /tn "NexusInstitutional24x7"
echo.
echo To delete the task:
echo   schtasks /delete /tn "NexusInstitutional24x7" /f
echo.
echo To manually run 24/7 monitor in console:
echo   python run_24_7.py --mode backtest --asset-class multi --venues 235
echo.
pause
