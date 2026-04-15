@echo off
chcp 65001 >nul
echo ================================================
echo Drone Gesture Control - AirSim Version
echo ================================================
echo.
echo Checking if AirSim is running...
echo.

REM Check if AirSim is running
tasklist /FI "WINDOWTITLE eq Blocks" 2>nul | find "Blocks.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] AirSim is not running!
    echo.
    echo Please start AirSim simulator first:
    echo   Double click: Blocks.exe
    echo   Note: Blocks.exe is usually in the AirSim installation directory
    echo.
    echo Press any key to continue (if AirSim is already running)...
    pause >nul
)

echo.
echo Starting gesture control program...
echo.

cd /d "%~dp0"
python main_airsim.py

echo.
echo ================================================
echo Program exited
pause
