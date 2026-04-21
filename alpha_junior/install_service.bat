@echo off
REM Install Alpha Junior as Windows Service (24/7 with auto-start)
REM Run as Administrator!

echo ==========================================
echo   Installing Alpha Junior as Service
echo ==========================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run as Administrator!
    echo Right-click - Run as administrator
    pause
    exit /b 1
)

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PYTHON_PATH=C:\Program Files\Python311\python.exe"

if not exist "%PYTHON_PATH%" (
    for /f "tokens=*" %%i in ('where python') do set "PYTHON_PATH=%%i"
)

echo Project: %PROJECT_DIR%
echo Python:  %PYTHON_PATH%
echo.

REM Create service using sc command
echo [1/3] Creating Windows Service...
sc create AlphaJunior binPath= "\"%PYTHON_PATH%\" \"%PROJECT_DIR%\service_runner.py\"" start= auto displayname= "Alpha Junior Fund Platform"
if %errorlevel% neq 0 (
    echo Service might already exist, continuing...
)

echo [2/3] Configuring Service...
sc description AlphaJunior "Institutional Fund Management Platform - Runs 24/7"
sc failure AlphaJunior reset= 86400 actions= restart/5000/restart/5000/restart/5000

echo [3/3] Starting Service...
net start AlphaJunior
if %errorlevel% neq 0 (
    echo Could not start service. Check logs.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   SUCCESS! Alpha Junior is now 24/7!
echo ==========================================
echo.
echo Service Status:
sc query AlphaJunior | findstr "STATE RUNNING"
echo.
echo Access: http://localhost:5000
echo.
echo Commands:
echo   net start AlphaJunior    - Start service
echo   net stop AlphaJunior     - Stop service
echo   sc delete AlphaJunior    - Uninstall
echo.
pause
