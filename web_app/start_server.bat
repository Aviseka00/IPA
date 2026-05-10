@echo off
cd /d "%~dp0backend"
echo Install deps once: pip install -r requirements.txt
echo Start MongoDB first (e.g. docker compose -f "%~dp0docker-compose.yml" up -d)
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
