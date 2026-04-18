@echo off
REM ===================================================================================
REM  NEXUS INSTITUTIONAL - HYBRID TRADING SYSTEM (NEWS + HFT)
REM ===================================================================================
REM  Starts the combined News Trading + High-Frequency Trading system
REM  Features:
REM    - News/sentiment trading (60-second event-driven)
REM    - HFT strategies (<100 microsecond latency)
REM    - Market-making, latency arbitrage, statistical arbitrage
REM    - Real-time portfolio tracking
REM ===================================================================================

setlocal enabledelayedexpansion

REM Set colors
set "GREEN=[92m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo %CYAN%╔════════════════════════════════════════════════════════════════════════════════╗%RESET%
echo %CYAN%║                  NEXUS INSTITUTIONAL - HYBRID TRADING SYSTEM                   ║%RESET%
echo %CYAN%║                    News/Sentiment Trading + High-Frequency Trading             ║%RESET%
echo %CYAN%╚════════════════════════════════════════════════════════════════════════════════╝%RESET%
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo %GREEN%✓ Python installed%RESET%

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo %GREEN%✓ Virtual environment activated%RESET%
) else (
    echo %YELLOW%! Virtual environment not found, using system Python%RESET%
)

echo.
echo %CYAN%System Configuration:%RESET%
echo.

REM Parse command line arguments
set "MODE=paper"
set "DURATION="
set "NEWS=enabled"
set "HFT=enabled"
set "LOG_LEVEL=INFO"

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--mode" (
    set "MODE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--duration" (
    set "DURATION=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--news" (
    set "NEWS=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--hft" (
    set "HFT=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--log-level" (
    set "LOG_LEVEL=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:args_done
echo   Execution Mode:  %MODE%
echo   News Trading:    %NEWS%
echo   HFT Engine:      %HFT%
if not "%DURATION%"=="" (
    echo   Duration:        %DURATION% seconds
) else (
    echo   Duration:        Unlimited ^(Ctrl+C to stop^)
)
echo   Log Level:       %LOG_LEVEL%
echo.

REM Build command
set "CMD=python hybrid_trading.py"
set "CMD=!CMD! --mode %MODE%"
set "CMD=!CMD! --news %NEWS%"
set "CMD=!CMD! --hft %HFT%"
set "CMD=!CMD! --log-level %LOG_LEVEL%"
if not "%DURATION%"=="" (
    set "CMD=!CMD! --duration %DURATION%"
)

echo %YELLOW%Starting hybrid trading system...%RESET%
echo %CYAN%Command: !CMD!%RESET%
echo.

REM Run the system
%CMD%

REM Capture exit code
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% equ 0 (
    echo.
    echo %GREEN%✓ System shutdown cleanly%RESET%
) else (
    echo.
    echo %YELLOW%! System exited with code %EXIT_CODE%%RESET%
)

echo.
pause
exit /b %EXIT_CODE%
