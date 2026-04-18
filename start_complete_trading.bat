@echo off
REM ===================================================================================
REM  NEXUS COMPLETE TRADING SYSTEM - WINDOWS LAUNCHER
REM ===================================================================================
REM  Starts the unified trading system with actual order execution
REM  Features:
REM    - Paper trading account ($1M starting capital)
REM    - Real order execution with market impact
REM    - Live portfolio tracking
REM    - News-driven trading opportunities
REM    - HFT-ready execution engine
REM ===================================================================================

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════════════════════╗
echo ║                    NEXUS COMPLETE TRADING SYSTEM                              ║
echo ║                     PAPER TRADING - NOW ACTUALLY TRADING!                     ║
echo ╚════════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [OK] Python installed
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [!] Virtual environment not found, using system Python
)

echo.
echo System Configuration:
echo.

REM Parse arguments or use defaults
set "MODE=paper"
set "CAPITAL=1000000"
set "DURATION="
set "LOG_LEVEL=INFO"

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--mode" (
    set "MODE=%~2"
    shift & shift
    goto parse_args
)
if "%~1"=="--capital" (
    set "CAPITAL=%~2"
    shift & shift
    goto parse_args
)
if "%~1"=="--duration" (
    set "DURATION=%~2"
    shift & shift
    goto parse_args
)
if "%~1"=="--log-level" (
    set "LOG_LEVEL=%~2"
    shift & shift
    goto parse_args
)
shift
goto parse_args

:args_done
echo   Execution Mode:  %MODE%
echo   Starting Capital: $%CAPITAL%
if not "%DURATION%"=="" (
    echo   Duration:        %DURATION% seconds
) else (
    echo   Duration:        Unlimited ^(Ctrl+C to stop^)
)
echo   Log Level:       %LOG_LEVEL%
echo.

REM Build command
set "CMD=python complete_trading_system.py"
set "CMD=!CMD! --mode %MODE%"
set "CMD=!CMD! --capital %CAPITAL%"
set "CMD=!CMD! --log-level %LOG_LEVEL%"
if not "%DURATION%"=="" (
    set "CMD=!CMD! --duration %DURATION%"
)

echo ════════════════════════════════════════════════════════════════════════════════
echo Starting trading system...
echo Command: !CMD!
echo ════════════════════════════════════════════════════════════════════════════════
echo.

REM Run the system
%CMD%

REM Capture exit code
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% equ 0 (
    echo.
    echo [OK] Trading session completed successfully
) else (
    echo.
    echo [!] Trading exited with code %EXIT_CODE%
)

echo.
echo Check trading_session_report.json for detailed results
echo.
pause
exit /b %EXIT_CODE%
