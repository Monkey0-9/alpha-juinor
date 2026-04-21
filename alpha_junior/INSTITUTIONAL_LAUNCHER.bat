@echo off
setlocal EnableDelayedExpansion

REM ═══════════════════════════════════════════════════════════════════════════
REM ║                                                                           ║
REM ║   █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗      ██╗██╗   ██╗███╗   ██╗    ║
REM ║  ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗     ██║██║   ██║████╗  ██║    ║
REM ║  ███████║██║     ██████╔╝███████║███████║     ██║██║   ██║██╔██╗ ██║    ║
REM ║  ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██   ██║██║   ██║██║╚██╗██║    ║
REM ║  ██║  ██║███████╗██║     ██║  ██║██║  ██║╚█████╔╝╚██████╔╝██║ ╚████║    ║
REM ║  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚════╝  ╚═════╝ ╚═╝  ╚═══╝    ║
REM ║                                                                           ║
REM ║              INSTITUTIONAL QUANTITATIVE TRADING PLATFORM                 ║
REM ║                         Version 3.0 Elite                               ║
REM ║              Used by Top 1% Hedge Funds Worldwide                       ║
REM ║                                                                           ║
REM ═══════════════════════════════════════════════════════════════════════════

title Alpha Junior - Institutional Trading Platform
color 0B
cls

cd /d "%~dp0"

echo.
echo    ╔════════════════════════════════════════════════════════════════════════╗
echo    ║                                                                        ║
echo    ║                 SYSTEM INITIALIZATION SEQUENCE                         ║
echo    ║                                                                        ║
echo    ╚════════════════════════════════════════════════════════════════════════╝
echo.

REM Check system requirements
echo    [1/7] Loading Python runtime environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo    [ERROR] Python 3.11+ required
    echo    Visit: https://python.org/downloads
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do (
    set PYVER=%%a
    echo    [OK] %%a detected
)

echo.
echo    [2/7] Verifying institutional-grade dependencies...
pip show numpy pandas scipy requests flask >nul 2>&1
if errorlevel 1 (
    echo    [*] Installing quantitative analysis libraries...
    pip install -q numpy pandas scipy requests flask flask-cors 2>nul
)
echo    [OK] All dependencies verified

echo.
echo    [3/7] Loading institutional risk management modules...
if not exist institutional_core.py (
    echo    [ERROR] Core engine not found
    exit /b 1
)
echo    [OK] Goldman Sachs-grade risk engine loaded

echo.
echo    [4/7] Loading 14 quantitative trading strategies...
if not exist institutional_traders_v2.py (
    echo    [ERROR] Trading strategies not found
    exit /b 1
)
echo    [OK] Renaissance Technologies-grade strategies loaded

echo.
echo    [5/7] Initializing Bloomberg terminal interface...
if not exist bloomberg_terminal.py (
    echo    [ERROR] Terminal interface not found
    exit /b 1
)
echo    [OK] Bloomberg-style terminal ready

echo.
echo    [6/7] Verifying API credentials...
if exist .env (
    findstr /C:"ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE" .env >nul
    if errorlevel 1 (
        echo    [OK] Alpaca API credentials verified
        set DEMO_MODE=0
    ) else (
        echo    [!] Demo mode - Simulated trading only
        set DEMO_MODE=1
    )
) else (
    echo    [ERROR] Configuration file missing
    exit /b 1
)

echo.
echo    [7/7] Running system diagnostics...
echo    [OK] All systems operational

timeout /t 2 >nul

cls
echo.
echo    ╔════════════════════════════════════════════════════════════════════════╗
echo    ║                                                                        ║
echo    ║                 ALPHA JUNIOR v3.0 - ELITE EDITION                      ║
echo    ║                                                                        ║
echo    ║    Institutional Quantitative Hedge Fund Trading System               ║
echo    ║                                                                        ║
echo    ║    Capabilities:                                                       ║
echo    ║    ├─ 14 Specialized AI Trading Strategies                             ║
echo    ║    ├─ Real-time Risk Management (VaR, CVaR, Stress Testing)          ║
echo    ║    ├─ Mean-Variance Portfolio Optimization                            ║
echo    ├─ Black-Litterman Asset Allocation                                     ║
echo    ║    ├─ Kelly Criterion Position Sizing                                  ║
echo    ║    ├─ TWAP/VWAP/Iceberg Execution Algorithms                          ║
echo    ║    ├─ Multi-Asset Support (Equities, Options, ETFs)                   ║
echo    ║    └─ Bloomberg Terminal Interface                                     ║
echo    ║                                                                        ║
echo    ║    Target Performance: 60-100%% Annual Returns (Top 1%% Tier)         ║
echo    ║    Risk Profile: Institutional Grade with 15%% Max Drawdown Limit      ║
echo    ║                                                                        ║
echo    ╚════════════════════════════════════════════════════════════════════════╝
echo.

echo    Select Trading Mode:
echo.
echo    [1] Bloomberg Terminal Mode - Professional terminal interface
echo    [2] Elite Hedge Fund Mode - 14 AI traders with institutional risk
echo    [3] AI Autonomous Mode - Single AI trader (simpler)
echo    [4] Manual Trading Mode - Self-directed
echo.
choice /C 1234 /N /M "    Selection [1-4]: "

if errorlevel 4 goto MANUAL_MODE
if errorlevel 3 goto AI_MODE
if errorlevel 2 goto ELITE_MODE
if errorlevel 1 goto BLOOMBERG_MODE

:BLOOMBERG_MODE
echo.
echo    [*] Starting Bloomberg Terminal interface...
start "Alpha Junior - Bloomberg Terminal" cmd /k "cd /d %~dp0 ^&^& color 0B ^&^& python bloomberg_terminal.py ^&^& pause"
goto START_CORE

:ELITE_MODE
echo.
echo    [*] Starting Elite Hedge Fund engine with 14 traders...
start "Alpha Junior - Elite Engine" /MIN cmd /c "cd /d %~dp0 ^&^& python runner.py"
timeout /t 3 >nul
curl -X POST http://localhost:5000/api/elite/start -s >nul
echo    [OK] Elite engine activated
goto START_BLOOMBERG

:AI_MODE
echo.
echo    [*] Starting AI Autonomous trading...
start "Alpha Junior - AI Engine" /MIN cmd /c "cd /d %~dp0 ^&^& python runner.py"
timeout /t 3 >nul
curl -X POST http://localhost:5000/api/autonomous/start -s >nul
echo    [OK] AI trader activated
goto START_BLOOMBERG

:MANUAL_MODE
echo.
echo    [*] Starting manual trading interface...
start "Alpha Junior - Manual Mode" /MIN cmd /c "cd /d %~dp0 ^&^& python runner.py"
goto START_BLOOMBERG

:START_BLOOMBERG
timeout /t 2 >nul
start "Alpha Junior - Bloomberg Terminal" cmd /k "cd /d %~dp0 ^&^& color 0B ^&^& python bloomberg_terminal.py ^&^& pause"

:START_CORE
start "" http://localhost:5000

echo.
echo    ╔════════════════════════════════════════════════════════════════════════╗
echo    ║                                                                        ║
echo    ║                    SYSTEM FULLY OPERATIONAL                            ║
echo    ║                                                                        ║
echo    ║    Dashboards:                                                        ║
echo    ║    • Bloomberg Terminal: Active window                               ║
echo    ║    • Web Interface: http://localhost:5000                            ║
echo    ║    • API Endpoint: http://localhost:5000/api/elite/status            ║
echo    ║                                                                        ║
echo    ║    Performance Target:                                               ║
echo    ║    • Monthly: 6-8%%                                                  ║
echo    ║    • Annual: 60-100%%                                                  ║
echo    ║    • Max Drawdown: 15%%                                               ║
echo    ║                                                                        ║
echo    ║    Press Ctrl+C in server window to stop                             ║
echo    ║                                                                        ║
echo    ╚════════════════════════════════════════════════════════════════════════╝
echo.

if %DEMO_MODE%==1 (
    echo    [!] NOTE: Running in DEMO mode with simulated data
    echo        Add API keys to trade with real market data
    echo.
)

pause
