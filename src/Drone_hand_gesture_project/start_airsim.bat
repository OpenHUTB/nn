@echo off
chcp 65001 >nul
echo ================================================
echo AirSim 手势控制无人机 - 启动器
echo ================================================
echo.

REM 检查 AirSim 是否运行
echo [1/3] 检查 AirSim 模拟器...
tasklist /FI "WINDOWTITLE eq Blocks" 2>nul | find "Blocks.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  AirSim 未运行，正在启动...
    start "" "d:\机械学习\air\Blocks\WindowsNoEditor\Blocks.exe"
    timeout /t 5 /nobreak >nul
    echo ✅ AirSim 已启动，请等待加载完成...
) else (
    echo ✅ AirSim 已在运行
)

echo.
echo [2/3] 进入项目目录...
cd /d "%~dp0"
echo ✅ 当前目录：%CD%

echo.
echo [3/3] 启动手势控制程序...
echo ================================================
echo.

python main.py

echo.
echo ================================================
echo 程序已退出
pause
