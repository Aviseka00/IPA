"""IPA Analysis Platform — FastAPI entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from passlib.exc import PasswordSizeError

logger = logging.getLogger(__name__)

from app.config import WEB_APP_ROOT, ensure_dirs, settings
from app.database import close_db, get_db
from app.routers import analysis, ani, auth, compliance, files, models, review


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    db = get_db()
    await db.users.create_index("employee_id", unique=True)
    await db.batches.create_index([("employee_id", 1), ("created_at", -1)])
    await db.analysis_items.create_index([("batch_id", 1), ("employee_id", 1)])
    await db.review_audit.create_index([("batch_id", 1), ("created_at", -1)])
    await db.review_audit.create_index([("item_id", 1), ("created_at", -1)])
    await db.system_audit.create_index([("created_at", -1)])
    await db.system_audit.create_index([("employee_id", 1), ("created_at", -1)])
    await db.system_audit.create_index("log_id", unique=True)

    # By default, open Ani alongside IPA so labeling is always ready
    if settings.ani_autostart:
        try:
            import asyncio

            from app.services import ani_bridge

            result = await asyncio.to_thread(ani_bridge.ensure_ani_running)
            logger.info(
                "Ani autostart: %s (project=%s)",
                result.get("message"),
                result.get("project_dir"),
            )
        except Exception:
            logger.exception("Ani autostart failed — IPA will continue without Ani")

    yield
    await close_db()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = settings.api_prefix
app.include_router(auth.router, prefix=api)
app.include_router(ani.router, prefix=api)
app.include_router(models.router, prefix=api)
app.include_router(analysis.router, prefix=api)
app.include_router(review.router, prefix=api)
app.include_router(compliance.router, prefix=api)
app.include_router(files.router, prefix=api)


@app.exception_handler(PasswordSizeError)
async def password_size_handler(_request: Request, exc: PasswordSizeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Server error — see backend logs for details"},
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/ready")
async def ready() -> dict:
    """Readiness for load balancers / team deployments."""
    from app.services import ani_bridge

    mongo_ok = False
    try:
        await get_db().command("ping")
        mongo_ok = True
    except Exception as exc:
        logger.warning("Mongo ping failed: %s", exc)
    ani_root = ani_bridge.resolve_ani_root()
    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongo": mongo_ok,
        "ani_root": str(ani_root),
        "ani_version": ani_bridge.ani_version(),
        "ani_python_ok": (ani_root / ".venv" / "Scripts" / "python.exe").is_file(),
        "inference_max_concurrent": settings.inference_max_concurrent,
        "mongodb_max_pool_size": settings.mongodb_max_pool_size,
    }


FRONTEND_DIST = WEB_APP_ROOT / "frontend" / "dist"


def _frontend_file(path: str) -> Path | None:
    if not path or path.endswith("/"):
        path = path.rstrip("/") or "index.html"
    target = (FRONTEND_DIST / path).resolve()
    root = FRONTEND_DIST.resolve()
    if not str(target).startswith(str(root)):
        return None
    return target if target.is_file() else None


if FRONTEND_DIST.is_dir():
    @app.get("/", include_in_schema=False)
    async def serve_root() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi"):
            raise HTTPException(status_code=404, detail="Not Found")
        static = _frontend_file(full_path)
        if static is not None:
            return FileResponse(static)
        index = FRONTEND_DIST / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend not built")
        return FileResponse(index)
