@echo off
REM Alpha Junior - AI AUTONOMOUS MODE
REM Full automation with AI Brain scanning all stocks

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║       🤖 ALPHA JUNIOR - AI AUTONOMOUS TRADING MODE                  ║
echo ║              Scans ALL Stocks - Uses AI Brain                       ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo [INITIALIZING SYSTEM...]
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found! Please install Python 3.11+
    pause
    exit /b 1
)
echo [OK] Python detected

REM Install dependencies
echo [1] Installing AI dependencies...
pip install -q flask flask-cors requests numpy 2>nul
echo [OK] Dependencies ready

REM Check .env
echo [2] Checking API configuration...
if not exist .env (
    echo [X] .env file not found!
    pause
    exit /b 1
)

findstr /C:"ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE" .env >nul
if not errorlevel 1 (
    echo.
    echo [!] WARNING: Alpaca API keys not configured!
    echo [!] Get your keys from: https://alpaca.markets/
    echo [!] Edit .env file and replace YOUR_SECRET_KEY_HERE
    echo.
    choice /C YN /M "Continue anyway (will not trade live)"
    if errorlevel 2 exit /b 1
)
echo [OK] Configuration checked

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                    🤖 AI BRAIN CAPABILITIES                         ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo What the AI Brain does:
echo   • Analyzes 100+ stocks every 5 minutes
echo   • Calculates momentum, RSI, trend, volatility
echo   • Scores each stock 0-100 (higher = better opportunity)
echo   • Auto-buys stocks scoring 75+ (strong buy signal)
echo   • Auto-sells positions scoring below 40
echo   • Stop-loss at -8%%, take-profit at +20%%
echo   • Manages up to 20 simultaneous positions
echo   • Targets: 50-60%% annual returns
echo.
echo Stock Universe: Tech, EV, Fintech, Biotech, Meme stocks, ETFs
echo Examples: AAPL, NVDA, TSLA, AMD, PLTR, COIN, GME, AMC, SPY, QQQ
echo.

echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                     OPENING DASHBOARDS                              ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo Launching:
echo   Window 1: AI Server + Trading Engine
echo   Window 2: Live Trade Monitor
echo   Browser:   Trading Interface
echo.

REM Start server with autonomous mode
echo [3] Starting AI Server...
start "Alpha Junior - AI SERVER" cmd /k "cd /d %~dp0 ^&^& color 0A ^&^& echo [AI Mode Active] ^&^& echo Starting Autonomous Trading System... ^&^& python runner.py ^&^& echo. ^&^& echo Server stopped. Press any key to exit. ^&^& pause"

echo [4] Waiting for server startup...
timeout /t 5 /nobreak >nul

REM Start dashboard
echo [5] Starting Live Monitor...
start "Alpha Junior - LIVE MONITOR" cmd /k "cd /d %~dp0 ^&^& color 0E ^&^& python monitor_dashboard.py ^&^& echo. ^&^& echo Monitor closed. ^&^& pause"

REM Open browser
echo [6] Opening browser interface...
timeout /t 2 /nobreak >nul
start http://localhost:5000

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                    ✅ SYSTEM FULLY RUNNING!                         ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo WHAT'S HAPPENING NOW:
echo   The AI Brain is scanning the entire stock market...
echo   It will automatically find and trade the best opportunities.
echo.
echo QUICK COMMANDS:
echo   Start AI Bot:  curl -X POST http://localhost:5000/api/autonomous/start
echo   Stop AI Bot:   curl -X POST http://localhost:5000/api/autonomous/stop
echo   Check Status:  curl http://localhost:5000/api/autonomous/status
echo   Brain Analyze: curl http://localhost:5000/api/brain/analyze
echo.
echo DASHBOARDS:
echo   Main Interface: http://localhost:5000
echo   Account Info:   http://localhost:5000/api/trading/account
echo   Positions:      http://localhost:5000/api/trading/positions
echo.
echo PAPER TRADING:
echo   You have $100,000 fake money to practice with.
echo   All trades use real market prices.
echo.
echo ⚠️  IMPORTANT:
echo   Keep the SERVER window open! This runs the AI.
echo   Dashboard updates every 3 seconds automatically.
echo   Press Ctrl+C in server window to stop.
echo.
echo 🎯 TARGET: 50-60%% Annual Returns
echo    Daily Target: ~0.15%% (compounds to 50-60%%/year)
echo    Monthly: ~4-5%%
echo.
pause
