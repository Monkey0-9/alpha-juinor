@echo off
REM ============================================================================
REM NEXUS INSTITUTIONAL - START LIVE PAPER TRADING MONITOR
REM ============================================================================
REM Real-time paper trading with news & event monitoring
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║          NEXUS INSTITUTIONAL - LIVE PAPER TRADING MONITOR                 ║
echo ║           Real-Time News & Event-Driven Trading System                    ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.
echo Configuration:
echo   Mode:           Paper Trading (No Real Money)
echo   Update Interval: 60 seconds
echo   Data Sources:   Bloomberg, Reuters, CNBC, MarketWatch, Seeking Alpha
echo   Risk Level:     Monitored (no execution without approval)
echo.
echo Features:
echo   ✓ Real-time news monitoring
echo   ✓ Sentiment analysis (TextBlob AI-powered)
echo   ✓ Event-driven trading signals
echo   ✓ Live portfolio monitoring
echo   ✓ 24/7 operational monitoring
echo.
echo Starting monitor... Press Ctrl+C to stop
echo.

REM Run the live trading monitor
python live_paper_trading.py --mode paper --log-level INFO

if errorlevel 1 (
    echo.
    echo ❌ Error running live monitor
    echo    Make sure you ran: setup_live_trading.bat first
    pause
)
