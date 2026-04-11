@echo off
REM =====================================================
REM Mini Quant Fund - One Command Launcher
REM =====================================================
REM This batch file runs the complete trading system
REM Usage: Just double-click this file or run: start.bat
REM =====================================================

echo.
echo  ================================================
echo   Mini Quant Fund - Automated Trading System
echo  ================================================
echo.
echo  Starting full system validation and trading...
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found! Please install Python 3.11+
    pause
    exit /b 1
)

REM Run the master script
python run_all.py

REM Pause to see results
echo.
echo  ================================================
echo  Press any key to exit...
echo  ================================================
pause >nul
