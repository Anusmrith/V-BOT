@echo off
title V-BOT Neural Terminal
cls
cd /d "%~dp0"

echo ===================================================
echo Launching V-BOT in Windows Terminal...
echo ===================================================

python vbot_ui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] V-BOT exited with an error code %ERRORLEVEL%.
    pause
)
