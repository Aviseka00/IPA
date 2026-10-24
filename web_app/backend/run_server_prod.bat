@echo off
REM Production-style API for a shared lab team (multi-worker).
REM Each analyst PC still runs Ani locally; this hosts the shared IPA API.
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt -q
pip install tifffile imagecodecs scikit-image scipy -q

set WORKERS=4
if defined UVICORN_WORKERS set WORKERS=%UVICORN_WORKERS%

echo Starting IPA API with %WORKERS% workers on 0.0.0.0:8010 ...
uvicorn app.main:app --host 0.0.0.0 --port 8010 --workers %WORKERS% --timeout-keep-alive 75 --limit-concurrency 2000
