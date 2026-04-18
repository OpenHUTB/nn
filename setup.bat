@echo off

REM Enable delayed expansion
setlocal enabledelayedexpansion

REM ========================================
REM Progress Tracking Configuration
REM ========================================
set "TOTAL_STEPS=11"
set "CURRENT_STEP=0"
set "SKIPPED_COUNT=0"
set "DOWNLOADED_COUNT=0"
set "ERROR_COUNT=0"

REM Record start time
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a "START_H=%%a, START_M=1%%b-100, START_S=1%%c-100, START_TOTAL=START_H*3600+START_M*60+START_S"
)

REM Get script directory
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM ========================================
REM Progress Display Function
REM ========================================
:show_progress
set /a "CURRENT_STEP+=1"
set /a "PERCENT=CURRENT_STEP*100/TOTAL_STEPS"

REM Calculate elapsed time
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a "CUR_H=%%a, CUR_M=1%%b-100, CUR_S=1%%c-100, CUR_TOTAL=CUR_H*3600+CUR_M*60+CUR_S"
)
set /a "ELAPSED=CUR_TOTAL-START_TOTAL, ELAPSED_M=ELAPSED/60, ELAPSED_S=ELAPSED%%60"

REM Estimate remaining time (simple average)
if !CURRENT_STEP! gtr 1 (
    set /a "AVG_TIME=ELAPSED/(CURRENT_STEP-1), REMAINING=(TOTAL_STEPS-CURRENT_STEP)*AVG_TIME"
    set /a "REMAINING_M=REMAINING/60, REMAINING_S=REMAINING%%60"
) else (
    set "REMAINING_M=--" & set "REMAINING_S=--"
)

REM Build progress bar
set /a "BAR_SIZE=PERCENT/2"
set "PROGRESS_BAR="
for /l %%i in (1,1,!BAR_SIZE!) do set "PROGRESS_BAR=!PROGRESS_BAR!#"
for /l %%i in (!BAR_SIZE!+1,1,50) do set "PROGRESS_BAR=!PROGRESS_BAR! "

REM Display progress
echo.
echo ================================================================================
echo   [!PROGRESS_BAR!] !PERCENT!%%   Step !CURRENT_STEP!/%TOTAL_STEPS%
echo   Current: %~1
echo   Elapsed: !ELAPSED_M!m !ELAPSED_S!s  ^|  ETA: !REMAINING_M!m !REMAINING_S!s
echo ================================================================================
echo.

goto :eof

REM ========================================
REM Step Result Function
REM ========================================
:step_result
if "%~1"=="skip" (
    set /a "SKIPPED_COUNT+=1"
    echo   [SKIP] %~2
) else if "%~1"=="download" (
    set /a "DOWNLOADED_COUNT+=1"
    echo   [DOWNLOAD] %~2
) else if "%~1"=="error" (
    set /a "ERROR_COUNT+=1"
    echo   [ERROR] %~2
) else (
    echo   [OK] %~2
)
echo.
goto :eof

REM ========================================
REM Summary Function
REM ========================================
:show_summary
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a "END_H=%%a, END_M=1%%b-100, END_S=1%%c-100, END_TOTAL=END_H*3600+END_M*60+END_S"
)
set /a "TOTAL_TIME=END_TOTAL-START_TOTAL, TM=TOTAL_TIME/60, TS=TOTAL_TIME%%60"

echo.
echo ================================================================================
echo                         SETUP SUMMARY
echo ================================================================================
echo   Total Steps:        !CURRENT_STEP!/%TOTAL_STEPS%
echo   Total Time:         !TM!m !TS!s
echo   Skipped:            !SKIPPED_COUNT!
echo   Downloaded:         !DOWNLOADED_COUNT!
echo   Errors:             !ERROR_COUNT!
echo ================================================================================
echo.
goto :eof

REM Set virtual environment and Python paths
set "VENV_PATH=%PROJECT_ROOT%\dependencies\prerequisites\miniconda3\envs\hutb_3.10"
set "PYTHON_EXE=%VENV_PATH%\python.exe"

REM Set CarlaUE4.exe path
set "CARLA_EXE=%PROJECT_ROOT%\hutb\CarlaUE4.exe"

REM Set numpy_tutorial.py path
set "MAIN_AI_PY=%PROJECT_ROOT%\src\chap01_warmup\numpy_tutorial.py"

REM Display setup header
echo.
echo ================================================================================
echo                    Neural Network Setup Script
echo ================================================================================
echo   Project Root: !PROJECT_ROOT!
echo   Python Path:  %PYTHON_EXE%
echo   Carla Path:   %CARLA_EXE%
echo ================================================================================
echo.

REM ========================================
REM Step 1: Check/Download hutb_downloader
REM ========================================
call :show_progress "Checking hutb_downloader.exe"

if not exist "%PROJECT_ROOT%\hutb_downloader.exe" (
    echo   Downloading hutb_downloader.exe from Gitee...
    curl -L -o "hutb_downloader.exe" "https://gitee.com/OpenHUTB/sw/releases/download/up/hutb_downloader.exe"
    if exist "%PROJECT_ROOT%\hutb_downloader.exe" (
        call :step_result "download" "hutb_downloader.exe"
    ) else (
        call :step_result "error" "Failed to download hutb_downloader.exe"
    )
) else (
    call :step_result "skip" "hutb_downloader.exe already exists"
)

REM ========================================
REM Step 2: Check/Download dependencies
REM ========================================
call :show_progress "Checking dependencies directory"

REM 如果 dependencies 目录不存在，则下载
if not exist "%PROJECT_ROOT%\dependencies" (
    echo   Downloading dependencies repository...
    start /wait "" "%PROJECT_ROOT%\hutb_downloader.exe" --repository dependencies
    if exist "%PROJECT_ROOT%\dependencies" (
        call :step_result "download" "dependencies repository"
    ) else (
        call :step_result "error" "Failed to download dependencies"
    )
) else (
    call :step_result "skip" "dependencies directory already exists"
)

REM ========================================
REM Step 3: Kill existing processes
REM ========================================
call :show_progress "Checking for existing processes"

REM 如果之前存在模拟器进程（包括后台进程），则先杀掉
set "KILLED=0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :2000 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
    set /a "KILLED+=1"
)
if !KILLED! gtr 0 (
    call :step_result "ok" "Killed !KILLED! process(es) on port 2000"
) else (
    call :step_result "skip" "No processes found on port 2000"
)

REM ========================================
REM Step 4: Check/Download hutb directory
REM ========================================
call :show_progress "Checking hutb directory"

REM 如果 hutb 目录不存在，则下载
if not exist "%PROJECT_ROOT%\hutb" (
    echo   Downloading hutb repository...
    REM 调用 hutb_downloader.exe，等待执行完成
    start /wait "" "%PROJECT_ROOT%\hutb_downloader.exe"
    if exist "%PROJECT_ROOT%\hutb" (
        call :step_result "download" "hutb repository"
    ) else (
        call :step_result "error" "Failed to download hutb"
    )
) else (
    call :step_result "skip" "hutb directory already exists"
)

REM ========================================
REM Step 5: Extract 7zip
REM ========================================
call :show_progress "Checking 7zip prerequisites"

REM 为了解压miniconda3
if not exist "dependencies\prerequisites\7zip" (
    echo   Extracting 7zip...
    powershell -Command "Expand-Archive -Path 'dependencies\prerequisites\7zip.zip' -DestinationPath 'dependencies\prerequisites\' -Force" || exit /b
    call :step_result "ok" "7zip extracted successfully"
) else (
    call :step_result "skip" "7zip folder already exists"
)

REM ========================================
REM Step 6: Extract miniconda3
REM ========================================
call :show_progress "Checking miniconda3 prerequisites"

if not exist "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3\" (
    echo   Extracting miniconda3 (this may take a while)...
    echo "%PROJECT_ROOT%\dependencies\prerequisites\7zip\7z.exe" x "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3.zip" -o"%PROJECT_ROOT%\dependencies\prerequisites\" -y >nul
    "%PROJECT_ROOT%\dependencies\prerequisites\7zip\7z.exe" x "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3.zip" -o"%PROJECT_ROOT%\dependencies\prerequisites\" -y >nul
    if exist "%PROJECT_ROOT%\dependencies\prerequisites\miniconda3\" (
        call :step_result "ok" "miniconda3 extracted successfully"
    ) else (
        call :step_result "error" "Failed to extract miniconda3"
    )
) else (
    call :step_result "skip" "miniconda3 folder already exists"
)

REM ========================================
REM Step 7: Validate Python environment
REM ========================================
call :show_progress "Validating Python environment"

REM Define port and URL to check
for /f "tokens=16" %%i in ('ipconfig ^|find /i "ipv4"') do set host_ip=%%i
echo   Detected IP: %host_ip%
set "PORT=3000"
set "CHECK_URL=http://%host_ip%:%PORT%"

REM Maximum wait time in seconds for main_ai.py to start
set "MAX_WAIT=60"

REM Wait time after starting CarlaUE4.exe
set "POST_CARLA_WAIT=3"

REM Check if virtual environment exists
if not exist "%VENV_PATH%" (
    call :step_result "error" "Virtual environment not found at %VENV_PATH%"
    pause
    exit /b 1
)
call :step_result "ok" "Virtual environment found"

REM Check if Python interpreter exists
if not exist "%PYTHON_EXE%" (
    call :step_result "error" "Python interpreter not found at %PYTHON_EXE%"
    pause
    exit /b 1
)
call :step_result "ok" "Python interpreter found"

REM Check if numpy_tutorial.py exists
if not exist "%MAIN_AI_PY%" (
    call :step_result "error" "numpy_tutorial.py not found at %MAIN_AI_PY%"
    pause
    exit /b 1
)
call :step_result "ok" "numpy_tutorial.py found"

REM ========================================
REM Step 8: Display Python version
REM ========================================
call :show_progress "Initializing Python environment"

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

REM ========================================
REM Step 9: Start numpy_tutorial.py
REM ========================================
call :show_progress "Starting numpy_tutorial.py"

REM 1. First, run numpy_tutorial.py
echo   Starting numpy_tutorial.py in background...
start "numpy_tutorial" "%PYTHON_EXE%" "%MAIN_AI_PY%"
call :step_result "ok" "numpy_tutorial.py process started"

REM ========================================
REM Step 10: Wait for service ready
REM ========================================
call :show_progress "Waiting for service to be ready"

REM Wait for 5 seconds initially to give numpy_tutorial.py time to start
echo   Waiting for service startup (initial delay 5s)...
timeout /t 5 /nobreak >nul

REM If port is listening, try to get a successful HTTP response
curl -s -o NUL -w "%%{http_code}" "%CHECK_URL%" | findstr "200" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    call :step_result "ok" "numpy_tutorial.py is ready at %CHECK_URL%"
    goto :START_CARLA
)

:CHECK_TIMER
REM Increment wait count
set /a WAIT_COUNT=WAIT_COUNT+1

REM Check if maximum wait time exceeded
if !WAIT_COUNT! geq %MAX_WAIT% (
    echo   Warning: Maximum wait time exceeded (%MAX_WAIT% seconds)
    call :step_result "skip" "numpy_tutorial.py not fully ready (timeout)"
    goto :START_CARLA
)

REM Wait for 2 seconds before checking again
echo   Waiting for %PORT%... (Attempt !WAIT_COUNT! of %MAX_WAIT%)
timeout /t 2 /nobreak >nul
goto :WAIT_LOOP

:START_CARLA
REM ========================================
REM Step 11: Start CarlaUE4.exe
REM ========================================
call :show_progress "Starting CarlaUE4.exe"

REM 2. Then, start CarlaUE4.exe asynchronously
if exist "%CARLA_EXE%" (
    echo   Starting CarlaUE4.exe...
    start "CarlaUE4" "%CARLA_EXE%"
    call :step_result "ok" "CarlaUE4.exe started"
) else (
    call :step_result "skip" "CarlaUE4.exe not found, skipping startup"
)

REM Wait for specified time after starting CarlaUE4
timeout /t %POST_CARLA_WAIT% /nobreak >nul

REM ========================================
REM Display Setup Summary
REM ========================================
call :show_summary

REM Set custom prompt
prompt [hutb_3.10] $P$G

REM Keep terminal open
cmd /k