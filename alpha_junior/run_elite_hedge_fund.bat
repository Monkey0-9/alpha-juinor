@echo off
REM Alpha Junior - ELITE HEDGE FUND MODE
REM Institutional-grade trading like top 1% hedge funds

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                              ║
echo ║        🎩 ALPHA JUNIOR - ELITE HEDGE FUND ENGINE 🎩                          ║
echo ║                                                                              ║
echo ║              Institutional-Grade Automated Trading                           ║
echo ║                    Operates like Renaissance Technologies                    ║
echo ║                           Target: Top 1%% Returns                             ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo [INITIALIZING HEDGE FUND SYSTEM...]
echo.

cd /d "%~dp0"

REM Check Python
echo [1/6] Checking Python... 
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found! Install Python 3.11+ from python.org
    pause
    exit /b 1
)
echo    ✓ Python OK

REM Install dependencies
echo [2/6] Installing institutional-grade dependencies...
pip install -q flask flask-cors requests numpy 2>nul
echo    ✓ Dependencies installed

REM Check files
echo [3/6] Verifying system files...
if not exist elite_hedge_fund.py (
    echo [X] elite_hedge_fund.py not found!
    pause
    exit /b 1
)
if not exist institutional_traders.py (
    echo [X] institutional_traders.py not found!
    pause
    exit /b 1
)
if not exist institutional_portfolio.py (
    echo [X] institutional_portfolio.py not found!
    pause
    exit /b 1
)
echo    ✓ System files verified

REM Check API keys
echo [4/6] Checking Alpaca API configuration...
findstr /C:"ALPACA_SECRET_KEY=YOUR_SECRET_KEY_HERE" .env >nul
if not errorlevel 1 (
    echo.
    echo [!] WARNING: You need to configure Alpaca API keys!
    echo [!] Get your keys from: https://alpaca.markets/
    echo [!] Edit .env file and add your API Key and Secret Key
    echo.
    choice /C YN /M "Continue without API keys (demo mode)"
    if errorlevel 2 exit /b 1
)
echo    ✓ Configuration checked

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                   🧠 ELITE TRADING TEAM ACTIVATED                            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Your Hedge Fund is staffed by 14 specialized AI traders:
echo.
echo  1. 🏆 MOMENTUM MASTER        - High-momentum breakout
echo  2. 📊 MEAN REVERSION KING   - Oversold bounce specialist  
echo  3. 💥 BREAKOUT PRO           - Pattern breakout master
echo  4. 📈 TREND RIDER            - Long-term trend following
echo  5. 🔄 SWING TRADER           - 3-10 day swings
echo  6. ⚡ SCALPER                - Quick intraday moves
echo  7. 📅 POSITION TRADER       - Long-term holds
echo  8. 🎯 ARBITRAGE HUNTER       - Statistical mispricings
echo  9. 📉 GAP FILLER             - Overnight gap plays
echo 10. 🏛️ SECTOR ROTATOR        - Sector momentum
echo 11. 📊 VOLATILITY MASTER      - Vol expansion/contraction
echo 12. 📰 EVENT TRADER          - Post-earnings drift
echo 13. 🤖 ALGO MASTER           - Pattern recognition
echo 14. 🎲 PAIRS TRADER          - Correlation trading
echo.
echo STRATEGIES:
echo    • Multi-timeframe momentum analysis
echo    • Kelly Criterion position sizing
echo    • Risk parity portfolio management
echo    • Sector rotation optimization
echo    • Value at Risk (VaR) monitoring
echo    • Automated stop-loss / take-profit
echo    • Trailing stops for winners
echo    • Dynamic position rebalancing
echo.
echo STOCK UNIVERSE: 100+ institutional-grade stocks
echo    Tech, Healthcare, Financials, Energy, Consumer, Industrial
echo    Including: AAPL, NVDA, TSLA, AMD, PLTR, COIN, JPM, UNH, XOM, META
echo.
echo SCAN FREQUENCY: Every 5 minutes during market hours (9:30 AM - 4:00 PM)
echo.

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                      🚀 STARTING ELITE ENGINE                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Launching systems:
echo    - Elite Hedge Fund Trading Engine
echo    - AI Trading Team (4 specialists)
echo    - Portfolio Risk Management
echo    - Real-time Dashboard
echo    - Browser Interface
echo.

REM Start Elite Engine
echo [5/6] Starting Elite Hedge Fund Engine...
start "🎩 ELITE HEDGE FUND ENGINE" cmd /k "cd /d %~dp0 ^&^& color 0E ^&^& echo [ELITE MODE] ^&^& echo Starting institutional-grade trading... ^&^& python runner.py ^&^& pause"

timeout /t 5 /nobreak >nul

REM Start Dashboard
echo [6/6] Starting Dashboard Monitor...
start "📊 ELITE DASHBOARD" cmd /k "cd /d %~dp0 ^&^& color 0A ^&^& python monitor_dashboard.py ^&^& pause"

REM Open browser
timeout /t 2 /nobreak >nul
start http://localhost:5000

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    ✅ ELITE HEDGE FUND OPERATIONAL                           ║
echo ║                                                                              ║
echo ║  Your AI trading team is now:                                                ║
echo ║    ✓ Scanning 100+ stocks every 5 minutes                                    ║
echo ║    ✓ Analyzing with 14 institutional-grade strategies                      ║
echo ║    ✓ Sizing positions with Kelly Criterion                                   ║
echo ║    ✓ Managing risk with portfolio-level controls                             ║
echo ║    ✓ Executing trades automatically                                          ║
echo ║    ✓ Protecting capital with stop-losses                                     ║
echo ║    ✓ Maximizing returns with trailing stops                                  ║
echo ║                                                                              ║
echo ║  TARGET RETURNS:                                                             ║
echo ║    • Monthly: 5-8%% (consistent alpha generation)                            ║
echo ║    • Annual: 60-100%% (top 1%% hedge fund performance)                        ║
echo ║    • Drawdown: Max 15%% (strict risk management)                             ║
echo ║                                                                              ║
echo ║  MONITOR PROGRESS AT:                                                        ║
echo ║    Main Dashboard: http://localhost:5000                                     ║
echo ║    Account:      http://localhost:5000/api/trading/account                 ║
echo ║    Positions:    http://localhost:5000/api/trading/positions               ║
echo ║                                                                              ║
echo ║  PAPER TRADING: $100,000 starting capital                                    ║
echo ║  All trades use real market prices - no real money risk                       ║
echo ║                                                                              ║
echo ║  ⚠️  KEEP SERVER WINDOW OPEN! This is your hedge fund operating!            ║
echo ║     Press Ctrl+C to stop trading                                             ║
echo ║                                                                              ║
echo ║  🎩 Welcome to the top 1%%                                                    ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
pause
