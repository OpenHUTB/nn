@echo off
chcp 65001 >nul
title Drone Hand Gesture Control System

echo ==========================================
echo    Drone Hand Gesture Control System
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python first!
    pause
    exit /b 1
)
echo [SUCCESS] Python environment ready
echo.

echo [2/3] Checking dependencies...
python -c "import pygame" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    if exist requirements.txt (
        pip install -r requirements.txt
    ) else (
        echo [INFO] requirements.txt not found, skipping installation
    )
)
echo [SUCCESS] Dependencies ready
echo.

echo [3/3] Starting program...
echo.
echo Please select mode:
echo   1. Use Launcher (Recommended)
echo   2. Run New Simulation
echo   3. Run Old Simulation
echo   4. Run AirSim Version
echo   5. Run AirSim (with Drone Camera!)
echo   6. Open Config Editor
echo.
set /p choice=Enter your choice (1-6):

if "%choice%"=="1" (
    echo.
    echo Starting launcher...
    python launcher.py
) else if "%choice%"=="2" (
    echo.
    echo Starting new simulation...
    python main_v2.py
) else if "%choice%"=="3" (
    echo.
    echo Starting old simulation...
    python main.py
) else if "%choice%"=="4" (
    echo.
    echo Starting AirSim version...
    python main_airsim.py
) else if "%choice%"=="5" (
    echo.
    echo Starting AirSim with DRONE CAMERA!
    python main_airsim_camera.py
) else if "%choice%"=="6" (
    echo.
    echo Opening config editor...
    python config_ui.py
) else (
    echo.
    echo [ERROR] Invalid choice!
    pause
    exit /b 1
)

echo.
echo Program exited
pause
