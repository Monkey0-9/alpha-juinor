@echo off
REM ============================================================================
REM NEXUS ALPACA SETUP WIZARD (Windows)
REM ============================================================================
REM This script sets up Alpaca trading credentials interactively
REM

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo NEXUS ALPACA SETUP WIZARD
echo ============================================================================
echo.
echo This wizard will help you configure Alpaca trading credentials.
echo.
echo Prerequisites:
echo   1. Alpaca account (free signup at https://alpaca.markets/)
echo   2. API Key and Secret Key from Alpaca Dashboard
echo.
echo Selecting Paper or Live?
echo   - Paper: Free simulated trading (recommended for testing)
echo   - Live: Real money trading (requires account funding)
echo.

REM Optional: Run Python setup script
python setup_alpaca.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Setup failed. Make sure alpaca-trade-api is installed:
    echo   pip install alpaca-trade-api
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo SUCCESS! Credentials configured.
echo ============================================================================
echo.
echo Next steps:
echo.
echo 1. Test with Paper Trading (recommended):
echo    python complete_trading_system.py --mode paper --broker alpaca
echo.
echo 2. After validation, try Live Trading:
echo    python complete_trading_system.py --mode live --broker alpaca --capital 1000
echo.
echo DO NOT start with large amounts!
echo Recommended: Start with $1,000 for at least 1 week of validation.
echo.
pause
