@echo off
echo ========================================
echo    VERA - DR Detection Demo Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Dependencies not installed. Installing now...
    echo This may take 2-3 minutes...
    echo.
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Try manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [2/3] Starting VERA demo...
echo.
echo The browser will open automatically at http://localhost:8501
echo.
echo To stop the demo: Press Ctrl+C or close this window
echo.

echo [3/3] Launching...
streamlit run demo_app.py

pause
