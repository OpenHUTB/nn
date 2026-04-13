@echo off

REM Start Traffic Sign Recognition and Autonomous Driving Control System
REM Double-click this file to run

cd /d "%~dp0"

echo Starting Traffic Sign Recognition and Autonomous Driving Control System...
echo Please make sure CARLA simulator is running

REM Check if CARLA simulator is running
netstat -ano | findstr :2000 >nul
if %errorlevel% neq 0 (
    echo Warning: CARLA simulator not detected on port 2000
    echo Please start CARLA simulator first, then run this script
    echo CARLA simulator should be in the same directory as the project
    pause
    exit /b 1
)

echo CARLA simulator detected, starting main program...

REM Run main program
python main.py

if %errorlevel% neq 0 (
    echo Program failed to start, please check error messages
    pause
    exit /b 1
)

pause