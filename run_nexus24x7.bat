@echo off
REM Wrapper to start Nexus 24/7 using the project's venv Python
cd /d %~dp0
"%~dp0.venv\Scripts\python.exe" -u "%~dp0nexus_24_7.py"