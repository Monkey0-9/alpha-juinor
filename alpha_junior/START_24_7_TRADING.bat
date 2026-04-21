@echo off
cd /d "%~dp0"

:: Simple 24/7 Trading Launcher
title Alpha Junior - 24/7 Trading
color 0A

cls
echo.
echo ==========================================
echo   ALPHA JUNIOR - 24/7 TRADING SYSTEM
echo ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

:: Install deps if needed
echo [*] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [*] Installing required packages...
    pip install -q flask flask-cors requests numpy pandas scipy
)

:: Check API keys
echo [*] Checking API configuration...
findstr /C:"YOUR_SECRET_KEY_HERE" .env >nul
if not errorlevel 1 (
    echo [!] WARNING: Using DEMO mode
    echo     Add real Alpaca API keys to .env for live trading
    echo     Get keys: https://alpaca.markets/
    echo.
)

:: Kill any existing processes
echo [*] Cleaning up existing processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul

:: Start server
echo [*] Starting Alpha Junior server...
echo [*] This window must stay open!
echo.
echo ==========================================
echo   SERVER RUNNING - DO NOT CLOSE THIS WINDOW
echo ==========================================
echo.
echo Access: http://localhost:5000
echo.
echo To stop: Press Ctrl+C
echo.

:: Run the server with auto-restart
:loop
    python runner.py
    echo.
    echo [!] Server stopped - restarting in 5 seconds...
    echo [!] Press Ctrl+C twice to stop completely
    timeout /t 5 >nul
goto loop
