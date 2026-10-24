@echo off
REM Start IPA assay (backend + frontend). Ani autostarts from the bundled Ani\ folder.
cd /d "%~dp0web_app"
call start_app.bat
