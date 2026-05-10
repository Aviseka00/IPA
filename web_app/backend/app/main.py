"""
FastAPI web server: upload microscopy images, run infection segmentation, store results in MongoDB.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_EXTENSIONS,
    MODEL_CHECKPOINT,
    PROCESSED_DIR,
    UPLOAD_DIR,
    WEB_APP_ROOT,
    ensure_dirs,
)
from .database import find_job, insert_job, jobs_collection, utcnow
from .services.inference_service import InferenceService

ensure_dirs()

logger = logging.getLogger(__name__)

app = FastAPI(title="Cell infection analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = WEB_APP_ROOT / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

_inference: InferenceService | None = None


def get_inference() -> InferenceService:
    global _inference
    if _inference is None:
        if not MODEL_CHECKPOINT.is_file():
            raise RuntimeError(f"Model checkpoint missing: {MODEL_CHECKPOINT}")
        _inference = InferenceService(MODEL_CHECKPOINT)
    return _inference


def _sanitize_filename(name: str | None) -> str:
    if not name:
        return "image"
    base = Path(name).name
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", base)
    return safe[:200] if safe else "image"


def _safe_processed_path(filename: str) -> Path:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    p = (PROCESSED_DIR / filename).resolve()
    root = PROCESSED_DIR.resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


def _safe_upload_path(filename: str) -> Path:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    p = (UPLOAD_DIR / filename).resolve()
    root = UPLOAD_DIR.resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


@app.get("/")
async def root():
    index = WEB_APP_ROOT / "static" / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Place static/index.html or open /docs for API."}


@app.get("/sw.js")
def service_worker_stub():
    """Quiet 404s when the browser looks for a root-level service worker (cached PWAs, extensions)."""
    body = (
        "self.addEventListener('install',e=>self.skipWaiting());"
        "self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));"
    )
    return Response(content=body, media_type="application/javascript")


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    threshold: float | None = Query(None),
    min_area: int = Query(40, ge=1, le=500000),
    border_margin: int = Query(
        8,
        ge=0,
        le=512,
        description="Width of frame strip for edge-touching detection (small blobs there are dropped)",
    ),
    border_touch_keep_min_area: int = Query(
        500,
        ge=1,
        le=500000,
        description="Edge-touching blobs with area below this are artifacts; larger ones count (partial real cells)",
    ),
    clahe_clip: float = Query(
        2.0,
        ge=0,
        le=10,
        description="CLAHE on L channel; 0 disables contrast boost",
    ),
    use_watershed: bool = Query(
        False,
        description="Optional: split merged predictions via watershed (default off = count model-positive regions)",
    ),
    watershed_peak_distance: int = Query(18, ge=3, le=400),
    opening_ksize: int = Query(5, ge=0, le=31),
    blend_alpha: float = Query(
        0.52,
        ge=0.05,
        le=1.0,
        description="Opacity of colored instance fill on brightfield overlay",
    ),
    write_observation_panel: bool = Query(
        True,
        description="Also save side-by-side observation PNG (heatmap | masks | clean overlay)",
    ),
):
    """
    Upload one or more images. Single vs batch is inferred from file count (metadata only).
    Saves originals under uploads/test_images/, outputs under processed_images/.
    Default counting matches training (binary mask at threshold). For dense plates,
    enable watershed + tune watershed_peak_distance (~12–22), opening (5–9).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_id = str(uuid.uuid4())
    job_type = "single" if len(files) == 1 else "batch"
    infer = get_inference()

    results: list[dict] = []
    errors: list[dict] = []

    for idx, upload in enumerate(files):
        orig_name = upload.filename or f"file_{idx}"
        ext = Path(orig_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append({"filename": orig_name, "error": f"Unsupported type {ext}"})
            continue

        stem = Path(orig_name).stem
        safe_stem = _sanitize_filename(stem)[:80]
        stored_name = f"{job_id}_{idx}_{safe_stem}{ext}"
        stored_path = UPLOAD_DIR / stored_name

        content = await upload.read()
        if len(content) > 80 * 1024 * 1024:
            errors.append({"filename": orig_name, "error": "File too large (max 80MB)"})
            continue

        stored_path.write_bytes(content)

        out_name = f"{job_id}_{idx}_{safe_stem}_processed.png"
        out_path = PROCESSED_DIR / out_name

        try:
            meta = infer.process_image_file(
                stored_path,
                out_path,
                threshold=threshold,
                min_area_px=min_area,
                border_margin_px=border_margin,
                border_touch_keep_min_area_px=border_touch_keep_min_area,
                clahe_clip=clahe_clip,
                blend_alpha=blend_alpha,
                use_watershed=use_watershed,
                watershed_peak_distance=watershed_peak_distance,
                opening_ksize=opening_ksize,
                write_observation_panel=write_observation_panel,
            )
        except Exception as e:
            errors.append({"filename": orig_name, "error": str(e)})
            continue

        ritem = {
                "original_filename": orig_name,
                "stored_filename": stored_name,
                "stored_url": f"/api/files/upload/{stored_name}",
                "processed_filename": out_name,
                "count": meta["count"],
                "processed_url": f"/api/files/processed/{out_name}",
                "analysis_settings": {
                    "clahe_clip": meta.get("clahe_clip"),
                    "blend_alpha": meta.get("blend_alpha"),
                    "use_watershed": meta.get("use_watershed"),
                    "watershed_peak_distance": meta.get("watershed_peak_distance"),
                    "opening_ksize": meta.get("opening_ksize"),
                    "min_area_px": meta.get("min_area_px"),
                    "border_margin_px": meta.get("border_margin_px"),
                    "border_touch_keep_min_area_px": meta.get("border_touch_keep_min_area_px"),
                    "artifacts_marked": meta.get("artifacts_marked"),
                    "threshold": meta.get("threshold"),
                    "observation_saved": meta.get("observation_saved"),
                },
        }
        if meta.get("observation_filename"):
            ritem["observation_filename"] = meta["observation_filename"]
            ritem["observation_url"] = f"/api/files/processed/{meta['observation_filename']}"
        results.append(ritem)

    doc = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "completed" if results else "failed",
        "created_at": utcnow(),
        "file_count": len(files),
        "results": results,
        "errors": errors,
    }
    persistence_warning: str | None = None
    try:
        insert_job(doc)
    except Exception as e:
        logger.warning("MongoDB job insert failed (analysis completed): %s", e)
        persistence_warning = (
            "Job was not saved to the database. Start MongoDB if you need history — "
            f"details: {e}"
        )

    return {
        "job_id": job_id,
        "job_type": job_type,
        "status": doc["status"],
        "results": results,
        "errors": errors,
        "persistence_warning": persistence_warning,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    doc = find_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    doc.pop("_id", None)
    return doc


@app.get("/api/files/processed/{filename}")
def download_processed(filename: str):
    path = _safe_processed_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.get("/api/files/upload/{filename}")
def download_upload_file(filename: str):
    path = _safe_upload_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/health")
def health():
    try:
        jobs_collection().estimated_document_count()
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return {"status": "ok", "mongodb": mongo_ok, "model_exists": MODEL_CHECKPOINT.is_file()}
