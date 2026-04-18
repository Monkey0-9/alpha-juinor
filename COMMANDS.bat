@echo off
REM ============================================================================
REM QUICK REFERENCE - NEXUS TRADING COMMANDS
REM ============================================================================
REM This file shows all the commands to run your trading system

echo.
echo NEXUS TRADING SYSTEM - COMMAND REFERENCE
echo ============================================================================
echo.
echo SETUP:
echo   python setup_alpaca.py                    [Setup Alpaca credentials]
echo   python test_alpaca.py                     [Test Alpaca connection]
echo   pip install alpaca-trade-api              [Install Alpaca SDK]
echo.
echo PAPER TRADING (Local Simulation):
echo   python complete_trading_system.py --broker paper
echo   python complete_trading_system.py --broker paper --duration 3600
echo.
echo PAPER TRADING (With Real Alpaca Data):
echo   python complete_trading_system.py --broker alpaca --mode paper
echo   python complete_trading_system.py --broker alpaca --mode paper --duration 3600
echo.
echo LIVE TRADING (Real Money - Start Small!):
echo   python complete_trading_system.py --broker alpaca --mode live --capital 1000
echo   python complete_trading_system.py --broker alpaca --mode live --capital 5000 --duration 604800
echo.
echo DOCUMENTATION:
echo   ALPACA_TRADING_GUIDE.md                   [Complete setup guide]
echo   ALPACA_INTEGRATION_COMPLETE.md            [What was added]
echo   README_TRADING_NOW_ACTIVE.md              [System overview]
echo.
echo MONITORING:
echo   Check: trading_session_report.json        [After each session]
echo.
echo ============================================================================
echo.
echo REMEMBER:
echo   - Start with paper trading for 1-7 days
echo   - Use small amounts ($1K) for first real trade
echo   - Monitor daily P&L
echo   - Have stop-loss ready
echo.
pause
