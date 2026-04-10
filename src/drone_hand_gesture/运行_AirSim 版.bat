@echo off
chcp 65001 >nul
echo ================================================
echo 手势控制无人机 - AirSim 真实模拟器版
echo ================================================
echo.
echo 检查 AirSim 是否运行...
echo.

REM 检查 AirSim 是否运行
tasklist /FI "WINDOWTITLE eq Blocks" 2>nul | find "Blocks.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] AirSim 未运行！
    echo.
    echo 请先启动 AirSim 模拟器:
    echo   双击运行：d:\机械学习\air\Blocks\WindowsNoEditor\Blocks.exe
    echo.
    echo 按任意键继续（如果 AirSim 已启动）...
    pause >nul
)

echo.
echo 正在启动手势控制程序...
echo.

cd /d "%~dp0"
python main_airsim.py

echo.
echo ================================================
echo 程序已退出
pause
