@echo off
cd /d "%~dp0"
echo.
echo IPA Always-On Server Setup
echo ==========================
echo This will:
echo   - Build the frontend for production
echo   - Start the app on port 8010 (single server)
echo   - Register auto-start when you sign in to Windows
echo   - Try to disable sleep (if run as Administrator)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_always_on.ps1"
echo.
pause
