@echo off
REM Run Ani independently (without starting the IPA web app).
REM Upgrading Ani\ later is automatically used by IPA on next launch.
cd /d "%~dp0Ani"
if not exist "run_ani.bat" (
  echo Ani folder missing. Expected: %~dp0Ani
  pause
  exit /b 1
)
call run_ani.bat %*
