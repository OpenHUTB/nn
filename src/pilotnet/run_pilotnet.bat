@echo off

REM This script will run PilotNet in Anaconda environment

REM Try to find activate.bat directly
set "ACTIVATE_BAT="

REM Check common Anaconda installation locations
if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\Anaconda3\Scripts\activate.bat"
if exist "%ProgramData%\Anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%ProgramData%\Anaconda3\Scripts\activate.bat"
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\Miniconda3\Scripts\activate.bat"
if exist "%ProgramData%\Miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%ProgramData%\Miniconda3\Scripts\activate.bat"
if exist "D:\Anaconda\Scripts\activate.bat" set "ACTIVATE_BAT=D:\Anaconda\Scripts\activate.bat"

if not defined ACTIVATE_BAT (
    echo Anaconda activation script not found.
    echo Please check your Anaconda installation.
    pause
    exit /b 1
)

REM Create a temporary script with all commands
set "RUN_SCRIPT=%TEMP%\run_pilotnet_full.bat"

REM Write all commands to the script
echo @echo off > "%RUN_SCRIPT%"
echo echo Running PilotNet... >> "%RUN_SCRIPT%"
echo echo ====================== >> "%RUN_SCRIPT%"
echo echo Changing to D drive... >> "%RUN_SCRIPT%"
echo D: >> "%RUN_SCRIPT%"
echo echo Navigating to project directory... >> "%RUN_SCRIPT%"
echo cd "D:\nn\src\pilotnet" >> "%RUN_SCRIPT%"
echo echo Activating conda environment... >> "%RUN_SCRIPT%"
echo call "%ACTIVATE_BAT%" pilotnet >> "%RUN_SCRIPT%"
echo echo ====================== >> "%RUN_SCRIPT%"
echo echo Launching PilotNet... >> "%RUN_SCRIPT%"
echo echo ====================== >> "%RUN_SCRIPT%"
echo python main.py >> "%RUN_SCRIPT%"
echo echo ====================== >> "%RUN_SCRIPT%"
echo echo Press any key to exit... >> "%RUN_SCRIPT%"
echo pause >> "%RUN_SCRIPT%"

REM Run the script
cmd /k "call "%RUN_SCRIPT%""

REM Clean up
timeout /t 1 /nobreak >nul
del "%RUN_SCRIPT%"
