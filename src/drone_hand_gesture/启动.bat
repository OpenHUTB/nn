@echo off
chcp 65001 >nul
echo ==========================================
echo    无人机手势控制系统 - 一键启动
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python！
    pause
    exit /b 1
)
echo [成功] Python 环境就绪
echo.

echo [2/3] 检查依赖...
python -c "import pygame" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    if exist requirements.txt (
        pip install -r requirements.txt
    ) else (
        echo [提示] 未找到 requirements.txt，跳过安装
    )
)
echo [成功] 依赖就绪
echo.

echo [3/3] 启动程序...
echo.
echo 请选择启动模式:
echo   1. 使用启动器（推荐）
echo   2. 直接运行新版仿真
echo   3. 直接运行旧版仿真
echo   4. 打开配置编辑器
echo   5. AirSim仿真模式（带无人机摄像头）
echo.
set /p choice=请输入选项 (1-5):

if "%choice%"=="1" (
    echo.
    echo 正在启动启动器...
    python launcher.py
) else if "%choice%"=="2" (
    echo.
    echo 正在启动新版仿真...
    python main_v2.py
) else if "%choice%"=="3" (
    echo.
    echo 正在启动旧版仿真...
    python main.py
) else if "%choice%"=="4" (
    echo.
    echo 正在打开配置编辑器...
    python config_ui.py
) else if "%choice%"=="5" (
    echo.
    echo 正在启动 AirSim仿真模式（带无人机摄像头）...
    python main_airsim_camera.py
) else (
    echo.
    echo [错误] 无效选项！
    pause
    exit /b 1
)

echo.
echo 程序已退出
pause
