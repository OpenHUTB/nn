@echo off
cls

echo ============================================
echo    DRONE VISION NAVIGATION SYSTEM
echo ============================================
echo    FIXED VERSION - KeyboardInterrupt Handled
echo ============================================
echo.

echo Step 1: Checking simulator...
tasklist | findstr "AbandonedPark.exe" > nul
if errorlevel 1 (
    echo Simulator not running!
    echo Please run AbandonedPark.exe manually
    echo Waiting 20 seconds...
    timeout /t 20 /nobreak > nul
) else (
    echo ✓ Simulator is running
)

echo.
echo Step 2: Activating Python environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install airsim opencv-python numpy
)

echo.
echo Step 3: Checking Python packages...
python -c "import airsim; print('✓ AirSim installed')" 2>nul
if errorlevel 1 (
    echo Installing AirSim...
    pip install airsim
)

python -c "import cv2; print('✓ OpenCV installed')" 2>nul
if errorlevel 1 (
    echo Installing OpenCV...
    pip install opencv-python
)

echo.
echo Step 4: Starting drone system...
echo Note: Press Ctrl+C at any time to safely exit
echo.
python drone_vision_system_en_fixed.py

echo.
pause