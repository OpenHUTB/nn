@echo off
chcp 65001 > nul  REM 支持 UTF-8 中文显示
cls

echo ============================================
echo   无人机视觉导航系统启动脚本
echo ============================================
echo.

:: 检查 Python 是否可用
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或不在 PATH 中，请安装 Python 3.8 以上版本。
    pause
    exit /b 1
)

:: 检查模拟器
echo [1/4] 检查模拟器...
tasklist | findstr /i "AbandonedPark.exe" > nul
if errorlevel 1 (
    echo ⚠️ 未检测到 AbandonedPark.exe 进程。
    echo 请手动启动模拟器，然后按任意键继续...
    pause > nul
) else (
    echo ✓ 模拟器已在运行
)
echo.

:: 处理虚拟环境
echo [2/4] 配置 Python 虚拟环境...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ 虚拟环境已激活
) else (
    echo 正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo 正在安装依赖（使用 requirements.txt）...
    if exist requirements.txt (
        pip install -r requirements.txt
    ) else (
        pip install airsim opencv-python numpy
    )
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
    echo ✓ 依赖安装完成
)
echo.

:: 检查关键依赖
echo [3/4] 验证依赖...
python -c "import airsim, cv2, numpy" 2>nul
if errorlevel 1 (
    echo ⚠️ 部分依赖未正确安装，尝试重新安装...
    pip install --upgrade airsim opencv-python numpy
)
echo ✓ 依赖验证通过
echo.

:: 启动主程序
echo [4/4] 启动导航系统...
set MAIN_SCRIPT=drone_vision_system_en.py
if not exist %MAIN_SCRIPT% (
    echo ❌ 找不到主程序文件: %MAIN_SCRIPT%
    echo 请确认当前目录下存在该文件。
    pause
    exit /b 1
)

python %MAIN_SCRIPT%
if errorlevel 1 (
    echo ❌ 程序运行出错，请查看上方日志。
) else (
    echo ✓ 程序正常结束
)

echo.
pause