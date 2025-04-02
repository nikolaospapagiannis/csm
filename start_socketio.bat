@echo off
echo Starting CSM Voice Assistant with Socket.IO...
echo ==============================

REM Activate virtual environment
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Set required environment variables
set NO_TORCH_COMPILE=1
set CHUNK_SIZE=30
set MAX_AUDIO_LENGTH=2000

REM Start the application
echo Starting server...
echo The web interface will be available at: http://127.0.0.1:5000
python app_socketio.py
