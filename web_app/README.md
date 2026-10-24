# IPA Cell Analysis Platform

Production-style web app to evaluate microscopy images with all IPA models.

**Monorepo:** Ani v0.4.0 (human labeling) lives in `../Ani` in this same git repository. See [../REPO_LAYOUT.md](../REPO_LAYOUT.md).

## Stack

- **Backend:** FastAPI, MongoDB (Motor), JWT auth, per-user file storage
- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS
- **Models:** IPAV2, IPA3.0, IPA4 Phase1/Phase2 (YOLO), IPA4 Fusion
- **Labeling:** Ani v0.4.0 (bundled) — starts with IPA for FN/FP correction → retrain data

## Quick start

### 1. MongoDB

**Atlas (cloud):** set `MONGODB_URI` and `MONGODB_DB` in `web_app/backend/.env` (see `.env.example`).

**Local (optional):**

```bash
cd web_app
docker compose up -d
```

### 2. Backend

```bash
cd web_app/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install tifffile
```

**Plate scan `.tif` files require `tifffile`** — without it analysis fails with `No module named 'tifffile'`.

```bash
copy .env.example .env
```

Edit `.env` — set a strong `SECRET_KEY`.

From `web_app/backend`:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Or double-click `web_app\backend\run_server.bat`. API docs: http://127.0.0.1:8010/docs

**Port 8010** is used deliberately — ports 8000/8001 are often taken by other local APIs.

### 3. Frontend

```bash
cd web_app/frontend
npm install
npm run dev
```

Open http://localhost:5173

## Features

| Feature | Description |
|--------|-------------|
| Register / Sign in | Employee ID + password only |
| Analyze | Single or bulk upload, model picker, side-by-side collage + count |
| History | Dashboard by batch or by date |
| Storage | `web_app/storage/users/{employee_id}/batches/{batch_id}/` |
| Sign out | Clears JWT on client |

## API (prefix `/api`)

- `POST /auth/register` — `{ employee_id, password }`
- `POST /auth/login` — returns JWT
- `GET /models` — available models
- `POST /analysis/run` — multipart: `model_id`, `files[]`
- `GET /analysis/history?group_by=batch|date`
- `GET /analysis/batches/{id}`
- `GET /files/{employee_id}/{batch_id}/{folder}/{filename}?access_token=...`

## Production notes

- Set `SECRET_KEY` and restrict `CORS_ORIGINS`
- Use HTTPS reverse proxy (nginx)
- Run MongoDB with auth + backups
- Consider GPU for faster inference (`INFERENCE_DEVICE=0`)
- Build frontend: `npm run build` and serve `dist/` behind nginx or mount in FastAPI

## Folder layout

```
web_app/
  backend/app/     FastAPI application
  frontend/src/    React UI
  storage/users/   User uploads (gitignored)
  docker-compose.yml
```
