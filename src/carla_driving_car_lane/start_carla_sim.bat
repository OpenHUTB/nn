@echo off
chcp 65001 >nul 2>&1
color 0A
mode con: cols=80 lines=25
title CARLA AUTOPILOT

echo ==============================================
echo          CARLA 0.9.15 AUTOPILOT LAUNCHER
echo ==============================================
echo.

echo Starting CARLA Simulator...
start "" "D:\carla0.9.15\CarlaUE4.exe" -quality-level=Low

echo.
echo Waiting for CARLA server (15s)...
timeout /t 15 /nobreak >nul

echo.
echo Starting autopilot script...
echo.

"D:\carla_automatic1\.venv\Scripts\python.exe" "D:\carla_automatic1\automatic_control.py"

echo.
echo Done.
pause
