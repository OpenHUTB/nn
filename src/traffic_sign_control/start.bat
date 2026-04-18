@echo off

REM Start Traffic Sign Recognition and Autonomous Driving Control System
REM Double-click this file to run

echo Starting Traffic Sign Recognition and Autonomous Driving Control System...
echo Please make sure CARLA simulator is running

REM Change to script directory
cd /d "%~dp0"

echo Checking CARLA simulator status...
netstat -ano | findstr :2000

if %errorlevel% neq 0 (
    echo Warning: CARLA simulator not detected on port 2000
    echo Please start CARLA simulator first, then run this script again
    echo Note: CARLA simulator should be in the same directory as the project
    pause
    exit /b 1
)

echo CARLA simulator detected, starting main program...
echo Press any key to continue...
pause >nul

REM Run main program
echo Starting main.py...
python main.py

if %errorlevel% neq 0 (
    echo Error: Program started failed, please check error message
    pause
    exit /b 1
)

echo Program started successfully!
pause