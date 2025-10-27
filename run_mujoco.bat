chcp 65001
@echo off
setlocal enabledelayedexpansion


set MUJOCO_TEMP_FILE_DIR=mujoco-3.3.7-windows-x86_64.zip
set MUJOCO_REPO=https://github.com/google-deepmind/mujoco/releases/download/3.3.7/mujoco-3.3.7-windows-x86_64.zip

if not exist "%MUJOCO_TEMP_FILE_DIR%" (
    echo %FILE_N% Retrieving mujoco from %MUJOCO_REPO% ...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%MUJOCO_REPO%', '%MUJOCO_TEMP_FILE_DIR%')"
)

powershell -Command "Expand-Archive '%MUJOCO_TEMP_FILE_DIR%' -DestinationPath '.' -Force"

call .\bin\simulate.exe
