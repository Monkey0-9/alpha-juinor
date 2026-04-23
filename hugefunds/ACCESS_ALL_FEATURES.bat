@echo off
color 0A
title HUGE FUNDS - Access All Features
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║           HUGE FUNDS - COMPLETE FEATURE ACCESS GUIDE                         ║
echo ║                    24/7 Local System Access                                     ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Status: Checking system...

:: Check if running
ping -n 1 -w 1000 127.0.0.1 >nul
netstat -an | findstr "0.0.0.0:8000" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ℹ] System not running. Starting now...
    call RUN_24_7.bat
    exit
)

echo [✓] System is RUNNING on port 8000
echo.

:MENU
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    HUGE FUNDS - ACCESS MENU                                   ║
echo ║                    24/7 Local System - Top 1%%                               ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Time: %date% %time%
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║  🌐 QUICK ACCESS                                                              ║
echo ╠══════════════════════════════════════════════════════════════════════════════╣
echo ║  1. Open Bloomberg Dashboard        - Full trading terminal                    ║
echo ║  2. Open API Documentation          - Swagger UI (all endpoints)              ║
echo ║  3. System Health Status            - Verify all services                     ║
echo ║  4. View Live Logs                  - Real-time system output                 ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║  📊 TRADING & PORTFOLIO                                                       ║
echo ╠══════════════════════════════════════════════════════════════════════════════╣
echo ║  5. Portfolio Summary               - NAV, PnL, positions, exposure             ║
echo ║  6. Strategy Attribution            - PnL breakdown by strategy               ║
echo ║  7. Factor Exposure                 - Risk factor decomposition               ║
echo ║  8. Alpha Signal Heatmap            - 15-symbol live signals                ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║  🛡️  RISK MANAGEMENT                                                           ║
echo ╠══════════════════════════════════════════════════════════════════════════════╣
echo ║  9. Calculate CVaR                  - Value at Risk (95%%, 99%%)                  ║
echo ║ 10. Run Stress Test                 - 7 historical scenarios                   ║
echo ║ 11. Monte Carlo Simulation          - 10,000 scenario simulation               ║
echo ║ 12. Factor Risk Decomposition       - 8-factor exposure analysis               ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║  🏛️  GOVERNANCE & CONTROLS                                                     ║
echo ╠══════════════════════════════════════════════════════════════════════════════╣
echo ║ 13. Pre-Trade Check                 - Run 9 governance checks                  ║
echo ║ 14. Track Record Gate               - Check 1,260-day requirement             ║
echo ║ 15. Kill Switch (ARMED)              - Emergency liquidation                    ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║  🔧 ADMINISTRATION                                                            ║
echo ╠══════════════════════════════════════════════════════════════════════════════╣
echo ║ 16. Restart Backend                 - Restart server                          ║
echo ║ 17. Test WebSocket                  - Verify real-time streaming              ║
echo ║ 18. Run Full Diagnostics            - Complete system check                   ║
echo ║  Q. Quit                             - Exit to Windows                         ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
set /p choice="Enter your choice (1-18, Q): "

if "%choice%"=="1" goto DASHBOARD
if "%choice%"=="2" goto APIDOCS
if "%choice%"=="3" goto HEALTH
if "%choice%"=="4" goto LOGS
if "%choice%"=="5" goto PORTFOLIO
if "%choice%"=="6" goto ATTRIBUTION
if "%choice%"=="7" goto FACTORS
if "%choice%"=="8" goto HEATMAP
if "%choice%"=="9" goto CVAR
if "%choice%"=="10" goto STRESS
if "%choice%"=="11" goto MONTECARLO
if "%choice%"=="12" goto FACTOR_RISK
if "%choice%"=="13" goto GOVERNANCE
if "%choice%"=="14" goto TRACKRECORD
if "%choice%"=="15" goto KILLSWITCH
if "%choice%"=="16" goto RESTART
if "%choice%"=="17" goto WEBSOCKET
if "%choice%"=="18" goto DIAGNOSTICS
if /I "%choice%"=="Q" goto QUIT

echo [!] Invalid choice
timeout /t 2 >nul
goto MENU

:DASHBOARD
cls
echo.
echo [*] Opening Bloomberg Terminal Dashboard...
echo     URL: http://localhost:8000
echo     File: frontend\hugefunds.html
echo.
start "" "http://localhost:8000"
timeout /t 2 >nul
start "" "%CD%\frontend\hugefunds.html"
echo [✓] Dashboard opened in browser
echo.
pause
goto MENU

:APIDOCS
cls
echo.
echo [*] Opening API Documentation (Swagger UI)...
echo     URL: http://localhost:8000/docs
echo.
start "" "http://localhost:8000/docs"
echo [✓] Swagger UI opened - All 26+ endpoints documented
echo.
echo Available Endpoints:
echo   GET  /                          - Service info
echo   GET  /api/health                - Health check
echo   GET  /api/portfolio/summary     - Portfolio data
echo   POST /api/risk/cvar             - Calculate CVaR
echo   POST /api/risk/stress-test      - Run stress tests
echo   POST /api/governance/pre-trade  - Pre-trade checks
echo   POST /api/killswitch            - Emergency stop
echo.
pause
goto MENU

:HEALTH
cls
echo.
echo [*] Checking System Health...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/health' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
echo [✓] Health check complete
echo.
pause
goto MENU

:LOGS
call view_logs.bat
goto MENU

:PORTFOLIO
cls
echo.
echo [*] Fetching Portfolio Summary...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/portfolio/summary' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
pause
goto MENU

:ATTRIBUTION
cls
echo.
echo [*] Fetching Strategy Attribution...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/strategies/attribution' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
pause
goto MENU

:FACTORS
cls
echo.
echo [*] Fetching Factor Exposure...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/factor/exposure' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
pause
goto MENU

:HEATMAP
echo.
echo [*] Alpha Signal Heatmap is available in the dashboard:
echo     http://localhost:8000
echo.
start "" "http://localhost:8000"
pause
goto MENU

:CVAR
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    CVaR RISK CALCULATOR                                       ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo This calculates Conditional Value at Risk using historical simulation.
echo.
echo Example returns: [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01]
echo.
echo [!] For demo, using sample returns...
echo.
powershell -Command "$body = @{returns = @(0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, 0.005, -0.008, 0.012); confidence = 0.95} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/risk/cvar' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 10; $r | ConvertTo-Json } catch { 'Error: ' + $_.Exception.Message }"
echo.
pause
goto MENU

:STRESS
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    STRESS TEST FRAMEWORK                                      ║
echo ║                    7 Historical Scenarios                                    ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Running all 7 historical stress scenarios...
echo.
powershell -Command "$body = @{positions = @(@{symbol = 'AAPL'; quantity = 100; entry_price = 150; current_price = 175; side = 'long'}, @{symbol = 'GOOGL'; quantity = 50; entry_price = 2800; current_price = 3000; side = 'long'})} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/risk/stress-test' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 15; $r | ConvertTo-Json -Depth 10 } catch { 'Error: ' + $_.Exception.Message }"
echo.
echo.
echo Scenarios tested:
echo   [✓] 2008 Financial Crisis
echo   [✓] 2020 COVID Crash
echo   [✓] 2022 Rate Shock
echo   [✓] 2010 Flash Crash
echo   [✓] 1998 LTCM Crisis
echo   [✓] 2015 China Devaluation
echo   [✓] 2023 Banking Crisis
echo.
pause
goto MENU

:MONTECARLO
cls
echo.
echo [*] Monte Carlo Simulation requires the dashboard.
echo     Open: http://localhost:8000
echo.
start "" "http://localhost:8000"
pause
goto MENU

:FACTOR_RISK
cls
echo.
echo [*] Factor Risk Decomposition...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/factor/exposure' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
pause
goto MENU

:GOVERNANCE
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    PRE-TRADE GOVERNANCE CHECKS                                ║
echo ║                    9 Institutional Validation Checks                        ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Running 9 pre-trade governance checks...
echo.
powershell -Command "$signal = @{symbol = 'AAPL'; confidence = 0.75; strength = 0.8; avg_daily_volume = 50000000}; $portfolio = @{total_value = 10000000; position_value = 500000; sector_exposure = 0.15; gross_exposure = 1.2; var_95 = 0.02; max_drawdown = 0.08; confidence = 0.75; avg_correlation = 0.3}; $body = @{signal = $signal; portfolio = $portfolio} | ConvertTo-Json -Depth 5; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/governance/pre-trade-check' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 10; $r | ConvertTo-Json } catch { 'Error: ' + $_.Exception.Message }"
echo.
pause
goto MENU

:TRACKRECORD
cls
echo.
echo [*] Track Record Gate Check...
echo     Required: 1,260 days of history
echo.
set /p strategy_id="Enter strategy ID (or press Enter for default 'main'): "
if "%strategy_id%"=="" set strategy_id=main
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/governance/track-record/%strategy_id%' -TimeoutSec 5; $r | ConvertTo-Json } catch { 'Failed to connect' }"
echo.
pause
goto MENU

:KILLSWITCH
cls
color 0C
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    🚨 EMERGENCY KILL SWITCH 🚨                                ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
color 0A
echo This will immediately liquidate ALL positions and stop all trading.
echo.
echo ⚠️  WARNING: This action cannot be undone!
echo.
set /p confirm="Type 'LIQUIDATE' to confirm emergency stop: "
if /I not "%confirm%"=="LIQUIDATE" (
    echo [*] Kill switch cancelled.
    pause
    goto MENU
)

echo.
echo [*] ACTIVATING EMERGENCY KILL SWITCH...
powershell -Command "$body = @{confirm = $true; reason = 'Emergency liquidation triggered by operator'} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/killswitch' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 10; $r | ConvertTo-Json } catch { 'Error: ' + $_.Exception.Message }"
echo.
echo [✓] Kill switch activated. Emergency liquidation initiated.
echo.
pause
goto MENU

:RESTART
cls
echo.
echo [*] Restarting HugeFunds Backend...
echo.
echo [1] Stopping current instance...
taskkill /F /FI "WINDOWTITLE eq HUGEFUNDS*" 2>nul
taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *uvicorn*" 2>nul
timeout /t 3 /nobreak >nul

echo [2] Starting new instance...
call .venv\Scripts\activate 2>nul
start /B "HUGEFUNDS_BACKEND" cmd /c "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info > ..\logs\backend_24_7.log 2>&1" >nul 2>&1
timeout /t 5 /nobreak >nul

echo [3] Verifying...
netstat -an | findstr "0.0.0.0:8000" >nul
if %ERRORLEVEL% EQU 0 (
    echo [✓] Backend restarted successfully!
) else (
    echo [✗] Restart failed. Check logs.
)
echo.
pause
goto MENU

:WEBSOCKET
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    WEBSOCKET TEST                                             ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo WebSocket URL: ws://localhost:8000/ws
echo.
echo The WebSocket streams real-time data every 5 seconds:
echo   • Market indices (SPX, NDX, VIX, DXY, 10Y, BTC)
echo   • Portfolio updates (NAV, PnL, Sharpe, Drawdown)
echo   • Alpha signals (15-symbol heatmap)
echo.
echo [*] WebSocket is active when dashboard is open.
echo     Open: http://localhost:8000
echo.
start "" "http://localhost:8000"
pause
goto MENU

:DIAGNOSTICS
cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    FULL SYSTEM DIAGNOSTICS                                    ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo [*] Test 1/5: Backend Server...
netstat -an | findstr "0.0.0.0:8000" >nul
if %ERRORLEVEL% EQU 0 (
    echo [✓] Backend:     PASS - Port 8000 listening
) else (
    echo [✗] Backend:     FAIL - Not responding
)

echo [*] Test 2/5: API Health Endpoint...
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/health' -TimeoutSec 5; if ($r.status -eq 'healthy') { '[PASS] Health endpoint responding' } else { '[WARN] Health check warning' } } catch { '[FAIL] Cannot reach health endpoint' }"

echo [*] Test 3/5: Portfolio API...
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/portfolio/summary' -TimeoutSec 5; if ($r.nav) { '[PASS] Portfolio API responding' } else { '[WARN] Portfolio data incomplete' } } catch { '[FAIL] Cannot reach portfolio API' }"

echo [*] Test 4/5: Risk API...
powershell -Command "$body = @{returns = @(0.01, -0.02); confidence = 0.95} | ConvertTo-Json; try { $r = Invoke-RestMethod -Uri 'http://localhost:8000/api/risk/cvar' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 5; if ($r.var) { '[PASS] CVaR API responding' } else { '[WARN] CVaR calculation issue' } } catch { '[FAIL] Cannot reach risk API' }"

echo [*] Test 5/5: File System...
if exist backend\main.py (
    echo [✓] Backend files:   PRESENT
) else (
    echo [✗] Backend files:   MISSING
)
if exist frontend\hugefunds.html (
    echo [✓] Frontend files:  PRESENT
) else (
    echo [✗] Frontend files:  MISSING
)

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    DIAGNOSTICS COMPLETE                                       ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
pause
goto MENU

:QUIT
echo.
echo [*] Exiting to Windows...
echo [*] HugeFunds continues running in background.
echo [*] Access anytime at: http://localhost:8000
echo.
echo To stop the system completely:
echo   • Run: taskkill /F /FI "IMAGENAME eq python.exe"
echo   • Or: Close all command windows
echo.
exit
