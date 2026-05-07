chcp 65001
@echo off
setlocal enabledelayedexpansion

REM ========================================
REM MuJoCo 版本配置
REM ========================================
set MUJOCO_VERSION=3.3.7
set MUJOCO_DIR=mujoco-%MUJOCO_VERSION%-windows-x86_64
set MUJOCO_ZIP=%MUJOCO_DIR%.zip
set MUJOCO_REPO=https://github.com/google-deepmind/mujoco/releases/download/%MUJOCO_VERSION%/%MUJOCO_DIR%.zip
set MUJOCO_EXE=.\%MUJOCO_DIR%\bin\simulate.exe

REM ========================================
REM 版本检测函数
REM ========================================
:check_version
echo [INFO] Checking MuJoCo version %MUJOCO_VERSION%...

REM 检查可执行文件是否存在
if exist "%MUJOCO_EXE%" (
    echo [OK] MuJoCo %MUJOCO_VERSION% found at: %MUJOCO_DIR%
    goto :run_mujoco
)

REM 检查解压目录是否存在但可执行文件缺失
if exist "%MUJOCO_DIR%" (
    echo [WARN] Directory exists but simulate.exe missing, re-downloading...
    rmdir /s /q "%MUJOCO_DIR%" 2>nul
)

REM 检查是否需要下载
if not exist "%MUJOCO_ZIP%" (
    echo [INFO] Downloading MuJoCo %MUJOCO_VERSION%...
    echo [INFO] URL: %MUJOCO_REPO%
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MUJOCO_REPO%' -OutFile '%MUJOCO_ZIP%' -UseBasicParsing"
    if !errorlevel! neq 0 (
        echo [ERROR] Download failed! Please check network connection.
        exit /b 1
    )
    echo [OK] Download completed.
) else (
    echo [OK] Archive already exists: %MUJOCO_ZIP%
)

REM 解压
echo [INFO] Extracting %MUJOCO_ZIP%...
powershell -Command "Expand-Archive '%MUJOCO_ZIP%' -DestinationPath '.' -Force"
if !errorlevel! neq 0 (
    echo [ERROR] Extraction failed!
    exit /b 1
)

REM 再次验证
if not exist "%MUJOCO_EXE%" (
    echo [ERROR] simulate.exe not found after extraction!
    exit /b 1
)
echo [OK] Extraction completed.

REM ========================================
REM 运行 MuJoCo
REM ========================================
:run_mujoco
echo [INFO] Launching MuJoCo simulate...
call "%MUJOCO_EXE%"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to launch MuJoCo!
    exit /b 1
)

echo [INFO] MuJoCo closed.
exit /b 0
