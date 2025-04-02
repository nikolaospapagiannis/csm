@echo off
echo Starting CSM Voice Chat Assistant...
echo ==============================

REM Activate virtual environment
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Set required environment variable
set NO_TORCH_COMPILE=1

REM Start the application
echo Starting server...
echo The web interface will be available at: http://127.0.0.1:5000
python app.py
