@echo off
echo CSM Voice Chat Assistant Installer
echo ==============================

REM Check for Python installation
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    exit /b 1
)

echo Creating virtual environment...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create virtual environment.
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies.
    exit /b 1
)

echo.
echo Installation complete!
echo.
echo To start the application:
echo 1. Open Command Prompt in this directory
echo 2. Run: start.bat
echo.
echo Alternatively, you can run these commands manually:
echo.
echo   call .venv\Scripts\activate
echo   set NO_TORCH_COMPILE=1
echo   python app.py
echo.
echo Press any key to exit...
pause > nul
