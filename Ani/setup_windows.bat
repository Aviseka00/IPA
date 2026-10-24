@echo off
REM ---- Ani one-time setup for Windows (needs internet only this once) ----
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python was not found.
  echo  Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo  and TICK "Add python.exe to PATH" during installation. Then run this file again.
  pause
  exit /b 1
)
echo Creating a private Python environment in .venv ...
%PY% -m venv .venv || (echo Could not create environment & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .
if errorlevel 1 (echo Installation failed. Check your internet connection and try again. & pause & exit /b 1)
echo.
echo  ===  Ani is installed. Double-click run_ani.bat to start (run_demo.bat to practise). ===
pause
