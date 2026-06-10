@echo off

echo 启动 YOLO 目标检测演示...
echo ===============================
echo 正在检查环境...

rem 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: Python 未安装
    echo 请先安装 Python 3.7 或更高版本
    pause
    exit /b 1
)

echo Python 已安装

rem 检查依赖
python -c "import ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装 YOLO 依赖...
    pip install ultralytics
    if %errorlevel% neq 0 (
        echo 错误: 安装依赖失败
        pause
        exit /b 1
    )
)

echo 依赖检查完成

rem 运行演示脚本
cd /d "%~dp0"
echo 正在启动演示...
python run_yolo_demo.py

pause