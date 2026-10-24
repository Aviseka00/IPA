@echo off
REM ---- For maintainers: build dist\Ani\ and (if Inno Setup is installed) dist\Ani-Setup-x.y.z.exe ----
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat call setup_windows.bat
call .venv\Scripts\activate.bat
python -m pip install "pyinstaller>=6"
python -m PyInstaller packaging\ani.spec --noconfirm --distpath dist --workpath build
if errorlevel 1 (echo Build failed & pause & exit /b 1)
dist\Ani\Ani-console.exe --selftest
if errorlevel 1 (echo Self-test of the built app FAILED & pause & exit /b 1)
where iscc >nul 2>nul && iscc packaging\installer.iss
echo Done: dist\Ani\Ani.exe   (installer in dist\ if Inno Setup is installed)
pause
