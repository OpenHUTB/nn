@echo off
echo ================================================
echo Drone Gesture Control - AirSim Version
echo ================================================
echo.
echo Starting gesture control program...
echo.
echo Please make sure AirSim (Blocks.exe) is running!
echo.

cd /d "%~dp0"
python main_airsim.py

echo.
echo ================================================
echo Program exited
echo ================================================
pause
