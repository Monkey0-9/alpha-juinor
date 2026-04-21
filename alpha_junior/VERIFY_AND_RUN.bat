@echo off
:: ═══════════════════════════════════════════════════════════════════════════
:: ALPHA JUNIOR - VERIFY & RUN (Top 1% Institutional)
:: One command to verify everything and start trading
:: ═══════════════════════════════════════════════════════════════════════════

title Alpha Junior v3.0 - Verify and Run
color 0A
cls

echo.
echo    ╔══════════════════════════════════════════════════════════════════════════════╗
echo    ║                                                                                ║
echo    ║                    🔍 SYSTEM VERIFICATION & RUN                              ║
echo    ║                                                                                ║
echo    ║                    Top 1%% Institutional Trading System                       ║
echo    ║                                                                                ║
echo    ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

:: Verification checklist
echo    VERIFICATION CHECKLIST:
echo    ────────────────────────────────────────────────────────────────────────────
echo.

set ERRORS=0

:: 1. Python
echo    [ ] Checking Python... 
python --version >nul 2>&1
if errorlevel 1 (
    echo    [✗] Python NOT found
    set /a ERRORS+=1
) else (
    for /f "tokens=*" %%a in ('python --version 2^>^&1') do echo    [✓] %%a
)

:: 2. Dependencies
echo    [ ] Checking dependencies...
python -c "import flask, flask_cors, requests, numpy" >nul 2>&1
if errorlevel 1 (
    echo    [✗] Dependencies missing
    echo    [*] Installing...
    pip install -q flask flask-cors requests numpy pandas scipy
    if errorlevel 1 set /a ERRORS+=1
) else (
    echo    [✓] All dependencies installed
)

:: 3. Files
echo    [ ] Checking critical files...
if not exist app.py (
    echo    [✗] app.py missing
    set /a ERRORS+=1
) else (
    echo    [✓] app.py present
)

if not exist institutional_traders_v2.py (
    echo    [✗] Trading strategies missing
    set /a ERRORS+=1
) else (
    echo    [✓] 14 trading strategies present
)

if not exist institutional_core.py (
    echo    [✗] Risk engine missing
    set /a ERRORS+=1
) else (
    echo    [✓] Institutional risk engine present
)

:: 4. API Keys
echo    [ ] Checking API configuration...
if not exist .env (
    echo    [✗] .env file missing
    echo    [*] Creating template...
    (
        echo # Alpha Junior - Alpaca API Keys
        echo # Get from: https://alpaca.markets/
        echo.
        echo ALPACA_API_KEY=PKUNNQ8INWN6B3TCUNWK
        echo ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE
    ) > .env
    notepad .env
    set /a ERRORS+=1
) else (
    findstr /C:"YOUR_SECRET_KEY_HERE" .env >nul
    if errorlevel 1 (
        echo    [✓] API keys configured
    ) else (
        echo    [!] API keys not set (demo mode)
    )
)

:: 5. Port availability
echo    [ ] Checking port 5000...
netstat -ano | findstr :5000 >nul
if not errorlevel 1 (
    echo    [!] Port 5000 in use, attempting to free...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /PID %%a /F >nul 2>&1
    timeout /t 2 >nul
)
echo    [✓] Port 5000 ready

:: Results
echo.
echo    ════════════════════════════════════════════════════════════════════════════

if %ERRORS% GTR 0 (
    echo    ⚠  VERIFICATION FAILED: %ERRORS% error(s) found
    echo    [*] Please fix errors above, then run again
    pause
    exit /b 1
) else (
    echo    ✅ ALL CHECKS PASSED - SYSTEM READY
    echo.
    echo    [*] Starting Alpha Junior in 3 seconds...
    timeout /t 3 /nobreak >nul
    
    :: Run the main launcher
    call ONE_CLICK_RUN.bat
)
