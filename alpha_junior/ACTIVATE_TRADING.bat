@echo off
cd /d "%~dp0"

title Alpha Junior - Activate Trading Engine

echo.
echo ==========================================
echo   ACTIVATING ALPHA JUNIOR TRADING
echo ==========================================
echo.

echo [*] Checking server status...
curl -s http://localhost:5000/api/health >nul 2>&1
if errorlevel 1 (
    echo [!] Server not running!
    echo [*] Please start server first:
    echo     START_24_7_TRADING.bat
    pause
    exit /b 1
)

echo [+] Server is running
echo.
echo [*] Activating Elite Hedge Fund engine...
echo [*] Deploying 14 AI traders...
curl -s -X POST http://localhost:5000/api/elite/start >nul 2>&1
if errorlevel 1 (
    echo [!] Failed to activate. Retrying...
    timeout /t 3 >nul
    curl -s -X POST http://localhost:5000/api/elite/start >nul 2>&1
)

echo.
echo [+] Trading engine activated!
echo.
echo ==========================================
echo   14 AI TRADERS NOW ACTIVE
echo ==========================================
echo.
echo Trading Strategies:
echo   - Momentum, Mean Reversion, Breakout
echo   - Trend, Swing, Scalping, Position
echo   - Arbitrage, Gap, Sector, Volatility
echo   - Event, Algo, Pairs
echo.
echo Risk Management:
echo   - VaR monitoring: 2%% limit
echo   - Max drawdown: 15%% limit
echo   - Position sizing: Kelly Criterion
echo.
echo [*] First trades in 5-15 minutes (market hours)
echo [*] Monitor at: http://localhost:5000
echo.
pause
