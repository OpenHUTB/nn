@echo off

REM Enable delayed expansion
setlocal enabledelayedexpansion

REM Get script directory
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM Set virtual environment and Python paths
set "VENV_PATH=%PROJECT_ROOT%\dependencies\prerequisites\miniconda3\envs\hutb_3.10"
set "PYTHON_EXE=%VENV_PATH%\python.exe"

REM Set CarlaUE4.exe path
set "CARLA_EXE=%PROJECT_ROOT%\hutb\CarlaUE4.exe"

REM Set numpy_tutorial.py path
set "MAIN_AI_PY=%PROJECT_ROOT%\src\chap01_warmup\numpy_tutorial.py"

REM Define port and URL to check
for /f "tokens=16" %%i in ('ipconfig ^|find /i "ipv4"') do set host_ip=%%i
echo IP:%host_ip%
set "PORT=3000"
set "CHECK_URL=http://%host_ip%:%PORT%"

REM Maximum wait time in seconds for main_ai.py to start
set "MAX_WAIT=60"

REM Wait time after starting CarlaUE4.exe
set "POST_CARLA_WAIT=3"

if not exist "%PROJECT_ROOT%\hutb_downloader.exe" (
    curl -L -o "hutb_downloader.exe" "https://gitee.com/OpenHUTB/sw/releases/download/up/hutb_downloader.exe"
) else (
    echo hutb_downloader.exe already exists.
)

REM 如果 dependencies 目录不存在，则下载
if not exist "%PROJECT_ROOT%\dependencies" (
    echo dependencies directory not found. Downloading...
    start /wait "" "%PROJECT_ROOT%\hutb_downloader.exe" --repository dependencies
    echo Download and extraction dependencies completed.
) else (
    echo dependencies repository already exists.
)

REM 如果之前存在模拟器进程（包括后台进程），则先杀掉
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :2000') do taskkill /F /PID %%a

REM 如果 hutb 目录不存在，则下载
if not exist "%PROJECT_ROOT%\hutb" (
    echo hutb directory not found. Downloading...

    REM 调用 hutb_downloader.exe，等待执行完成
    start /wait "" "%PROJECT_ROOT%\hutb_downloader.exe"
    echo Download and extraction completed.
) else (
    echo hutb repository already exists.
    REM Check if CarlaUE4.exe exists
    if not exist "%CARLA_EXE%" (
        echo Warning: CarlaUE4.exe not found, skipping startup
        echo CarlaUE4.exe path: %CARLA_EXE%
    )
    start "CarlaUE4" "%CARLA_EXE%"
    
    REM Wait for specified time after starting CarlaUE4
    timeout /t %POST_CARLA_WAIT% /nobreak >nul
)

REM 为了解压miniconda3
if not exist "dependencies\prerequisites\7zip" (
    echo Unzipping 7zip ...
    powershell -Command "Expand-Archive -Path 'dependencies\prerequisites\7zip.zip' -DestinationPath 'dependencies\prerequisites\' -Force" || exit /b
) else (
    echo 7zip folder already exists.
)
if not exist "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3\" (
    echo Unzipping miniconda...
    echo "%PROJECT_ROOT%\dependencies\prerequisites\7zip\7z.exe" x "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3.zip" -o"%PROJECT_ROOT%\dependencies\prerequisites\" -y >nul
    "%PROJECT_ROOT%\dependencies\prerequisites\7zip\7z.exe" x "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3.zip" -o"%PROJECT_ROOT%\dependencies\prerequisites\" -y >nul
) else (
    echo miniconda3 folder already exists.
)

REM Check if virtual environment exists
if not exist "%VENV_PATH%" (
    echo Error: Virtual environment not found at %VENV_PATH%
    pause
    exit /b 1
)

REM Check if Python interpreter exists
if not exist "%PYTHON_EXE%" (
    echo Error: Python interpreter not found at %PYTHON_EXE%
    pause
    exit /b 1
)

REM Check if numpy_tutorial.py exists
if not exist "%MAIN_AI_PY%" (
    echo Error: numpy_tutorial.py not found at %MAIN_AI_PY%
    pause
    exit /b 1
)

REM Print activation information
echo Activating virtual environment...

REM Print Python version
echo Virtual environment activated successfully!
echo Python version:
%PYTHON_EXE% --version

echo Install hutb package:
REM 需要关闭代理，解决安装 whl 时的代理问题: WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))': /simple/msgpack-rpc-python/
REM 制作 Python 环境步骤：
REM dependencies/prerequisites/miniconda3/envs/hutb_3.10/python.exe -m pip install hutb\PythonAPI\carla\dist\hutb-2.9.16-cp310-cp310-win_amd64.whl
REM dependencies/prerequisites/miniconda3/envs/hutb_3.10/python.exe -m pip install fastapi uvicorn aiohttp fastmcp loguru

REM Set environment variables
set "PATH=%VENV_PATH%\Scripts;%VENV_PATH%;%PATH%"
set "VIRTUAL_ENV=%VENV_PATH%"

REM 1. First, run numpy_tutorial.py
echo Running numpy_tutorial.py...
start "numpy_tutorial" "%PYTHON_EXE%" "%MAIN_AI_PY%"

REM Wait for 5 seconds initially to give numpy_tutorial.py time to start
timeout /t 5 /nobreak >nul

REM If port is listening, try to get a successful HTTP response
curl -s -o NUL -w "%%{http_code}" "%CHECK_URL%" | findstr "200" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo numpy_tutorial.py is ready and responding at %CHECK_URL%!
    goto :START_CARLA
)

:CHECK_TIMER
REM Increment wait count
set /a WAIT_COUNT=WAIT_COUNT+1

REM Check if maximum wait time exceeded
if !WAIT_COUNT! geq %MAX_WAIT% (
    echo Warning: Maximum wait time exceeded (%MAX_WAIT% seconds)
    echo numpy_tutorial.py may not be fully ready, but continuing with CarlaUE4.exe startup...
    goto :START_CARLA
)

REM Wait for 2 seconds before checking again
echo Waiting for %PORT%... (Attempt !WAIT_COUNT! of %MAX_WAIT%)
timeout /t 2 /nobreak >nul
goto :WAIT_LOOP

:START_CARLA
REM 2. Then, start CarlaUE4.exe asynchronously
if exist "%CARLA_EXE%" (
    echo Existing CarlaUE4.exe...
)


REM Set custom prompt
prompt [hutb_3.10] $P$G

REM Keep terminal open
cmd /k