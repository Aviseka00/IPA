@echo off
setlocal
REM =====================================================================================
REM  Add AI to Ani (all CPU, no graphics card needed). Needs internet ONCE, about 2-3 GB.
REM    Cellpose 3  - auto-labels whole images, AI click
REM    micro-sam   - AI click with Segment-Anything for microscopy (needs Python 3.11+)
REM    YOLO        - runs YOLO-seg models you train (Colab notebook from Ani's export)
REM  Safe to run again: finished parts are skipped, failed parts are retried.
REM =====================================================================================
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Ani is not installed yet - running setup_windows.bat first...
  call setup_windows.bat
  if not exist .venv\Scripts\activate.bat exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [1/5] PyTorch, CPU version (about 250 MB)...
python -m pip install "torch>=2.5" torchvision || goto fail

echo.
echo [2/5] Cellpose 3...
python -m pip install "cellpose>=3.1,<4" || goto fail

echo.
echo [3/5] YOLO (ultralytics)...
python -m pip install ultralytics || goto fail

echo.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo [4/5] micro-sam SKIPPED: it needs Python 3.11 or newer.
  echo       To get it: install Python 3.12, delete the .venv folder, run setup_windows.bat and this file again.
  goto models
)
echo [4/5] micro-sam + MobileSAM...
REM MobileSAM is only on GitHub; installing from the zip download means Git is not needed.
python -m pip install micro_sam "mobile_sam @ https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip" || goto fail

:models
echo.
echo [5/5] Downloading the AI models (only this once)...
python -m ani --download-models
echo.
python -m ani --selftest
echo.
echo  ===  Done. Start Ani (run_ani.bat), open the AI tab and press Refresh.  ===
pause
exit /b 0

:fail
echo.
echo  Installation stopped. Check your internet connection and run install_ai.bat again
echo  (parts that already finished are kept).
pause
exit /b 1
