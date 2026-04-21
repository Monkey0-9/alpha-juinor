@echo off
REM Alpha Junior - Management Console
echo.
echo ==========================================
echo   Alpha Junior - Management Console
echo ==========================================
echo.
echo  [1] Start 24/7 Mode (Auto-restart on crash)
echo  [2] Start Service Mode (Runs without user logged in)
echo  [3] Add to Windows Startup
echo  [4] View Logs
echo  [5] Check Status
echo  [6] Stop Server
echo  [7] Install as Windows Service (Admin)
echo  [8] Uninstall Service (Admin)
echo  [9] Exit
echo.
set /p choice="Enter choice (1-9): "

if "%choice%"=="1" goto start_24_7
if "%choice%"=="2" goto start_service
if "%choice%"=="3" goto add_startup
if "%choice%"=="4" goto view_logs
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto stop_server
if "%choice%"=="7" goto install_service
if "%choice%"=="8" goto uninstall_service
if "%choice%"=="9" goto exit

goto end

:start_24_7
echo Starting 24/7 mode...
start "Alpha Junior 24/7" run_24_7.bat
echo Started in new window!
echo.
echo Access: http://localhost:5000
goto end

:start_service
echo Starting service mode...
net start AlphaJunior 2>nul
if %errorlevel% neq 0 (
    echo Service not installed. Run option 7 first.
) else (
    echo Service started!
    echo Access: http://localhost:5000
)
goto end

:add_startup
call add_to_startup.bat
goto end

:view_logs
if not exist logs\alpha_junior.log (
    echo No logs found yet.
    goto end
)
echo.
echo === Last 50 Log Lines ===
type logs\alpha_junior.log 2>nul | findstr /n "." | findstr "^[0-9]*:[4-5][0-9]" | tail -50
echo.
echo Full log: logs\alpha_junior.log
pause
goto end

:check_status
echo.
echo === Server Status ===
curl -s http://localhost:5000/api/health 2>nul
if %errorlevel%==0 (
    echo.
    echo.
    echo STATUS: RUNNING ^& HEALTHY
    echo URL: http://localhost:5000
) else (
    echo.
    echo STATUS: NOT RUNNING
    echo Start with option 1 or 2
)
goto end

:stop_server
echo.
echo Stopping all Alpha Junior processes...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Alpha Junior*" 2>nul
net stop AlphaJunior 2>nul
echo Stopped!
goto end

:install_service
echo.
echo Running service installer (Admin required)...
call install_service.bat
goto end

:uninstall_service
echo.
echo Uninstalling service...
net stop AlphaJunior 2>nul
sc delete AlphaJunior
echo Uninstalled!
goto end

:exit
goto end

:end
echo.
pause
