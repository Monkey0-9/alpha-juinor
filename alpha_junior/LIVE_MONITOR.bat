@echo off
cd /d "%~dp0"
title Alpha Junior - LIVE TERMINAL MONITOR
color 0B

cls
echo.
echo ==========================================
echo   ALPHA JUNIOR - LIVE TERMINAL MONITOR
echo ==========================================
echo.
echo This window shows EVERYTHING in real-time:
echo   - All trades
echo   - All positions
echo   - P/andL updates
echo   - AI analysis
echo   - Risk metrics
echo.
echo Starting monitor...
timeout /t 2 >nul

python terminal_monitor.py
