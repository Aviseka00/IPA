@echo off
REM Upgrade the bundled Ani used by IPA.
REM After this, IPA assay uses the new Ani automatically (same Ani\ path).
setlocal
cd /d "%~dp0"
echo.
echo IPA uses the Ani app in: %CD%\Ani
echo.
echo How to upgrade
echo --------------
echo 1. Close Ani if it is open.
echo 2. Replace or update files inside Ani\ with the new Ani release
echo    ^(keep your existing Ani\.venv if you only update source,
echo     or delete Ani\.venv and run setup again for a clean install^).
echo 3. This script will re-run Ani setup so the venv matches the new code.
echo.
set /p GO=Type Y and Enter to run Ani\setup_windows.bat now: 
if /I not "%GO%"=="Y" (
  echo Cancelled.
  exit /b 0
)
if not exist "Ani\setup_windows.bat" (
  echo Ani\setup_windows.bat not found.
  pause
  exit /b 1
)
cd Ani
call setup_windows.bat
echo.
echo Done. Restart IPA ^(run_ipa.bat^) or Open in Ani from the assay —
echo IPA will load this upgraded Ani automatically.
pause
