@echo off
chcp 65001 >nul
echo ================================================
echo Drone Gesture Control - Local Simulation
echo ================================================
echo.
echo Starting...
echo.

cd /d "%~dp0"
python main.py

echo.
echo ================================================
echo Program exited
pause
