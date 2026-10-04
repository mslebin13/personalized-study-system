@echo off
setlocal
cd /d "%~dp0"

title Personalized Study System - Localhost

echo ================================================
echo   Personalized Study Recommendation System
echo   Localhost Launcher
echo ================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo Python was not found.
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo IMPORTANT: enable "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment...
    %PY% -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists.
)

echo [2/3] Installing/updating required packages...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo [3/3] Starting the application...
echo.
echo Your website will open at http://127.0.0.1:5000
echo Press Ctrl+C in this window to stop the server.
echo.

start "" http://127.0.0.1:5000
"venv\Scripts\python.exe" app.py

pause
