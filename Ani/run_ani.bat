@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Ani is not installed yet. Double-click setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set QT_ENABLE_HIGHDPI_SCALING=1
python -m ani %*
if errorlevel 1 pause
