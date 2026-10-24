@echo off
echo Stopping stale IPA backend on port 8010...
for %%P in (8010) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
  )
)
timeout /t 2 /nobreak >nul

echo Preparing default Ani workspace...
if exist "%~dp0backend\.venv\Scripts\python.exe" (
  "%~dp0backend\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'%~dp0backend'); from app.services.ani_bridge import ensure_default_workspace; print(ensure_default_workspace())"
)

echo Starting Ani v0.4.0 (bundled at ..\Ani\)...
if exist "%~dp0..\Ani\run_ani.bat" (
  start "Ani" cmd /c "cd /d %~dp0..\Ani && call run_ani.bat --full %~dp0storage\ani_workspace"
) else (
  echo Ani not found — expected folder: IPA\Ani  ^(run Ani\setup_windows.bat once after clone^)
)

echo Starting backend on http://0.0.0.0:8010 ...
start "IPA Backend" cmd /k "cd /d %~dp0backend && call run_server.bat"

timeout /t 5 /nobreak >nul

echo Starting frontend (LAN: http://10.40.20.30:5173) ...
start "IPA Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Local:  http://localhost:5173
echo LAN:    http://10.40.20.30:5173
echo API docs: http://127.0.0.1:8010/docs
echo Ani opens automatically with IPA (labeling companion)
