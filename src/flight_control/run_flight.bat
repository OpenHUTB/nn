@echo off

:: Get current script directory
set SCRIPT_DIR=%~dp0

:: Navigate to project root directory
cd /d %SCRIPT_DIR%\..\..

:: Run main script
python src\flight_control\main.py

pause
