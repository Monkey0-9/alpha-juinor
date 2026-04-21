@echo off
REM Add Alpha Junior to Windows Startup (User level - no admin needed)

echo ==========================================
echo   Adding Alpha Junior to Startup
echo ==========================================
echo.

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "SHORTCUT_NAME=Alpha Junior 24_7"

REM Create startup shortcut
echo [1/2] Creating shortcut...

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\%SHORTCUT_NAME%.lnk'); $Shortcut.TargetPath = '%PROJECT_DIR%\run_24_7.bat'; $Shortcut.WorkingDirectory = '%PROJECT_DIR%'; $Shortcut.IconLocation = '%SystemRoot%\system32\shell32.dll, 21'; $Shortcut.Save()"

if %errorlevel% neq 0 (
    echo ERROR: Could not create shortcut
    pause
    exit /b 1
)

echo [2/2] Verifying...
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\%SHORTCUT_NAME%.lnk" (
    echo SUCCESS!
) else (
    echo WARNING: Shortcut may not have been created
)

echo.
echo ==========================================
echo   Alpha Junior will start automatically
echo   every time you log into Windows!
echo ==========================================
echo.
echo Location: %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
echo.
echo To remove: Delete the shortcut from that folder
echo.
pause
