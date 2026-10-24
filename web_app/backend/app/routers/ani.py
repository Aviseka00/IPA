"""Ani companion app — ensure running when IPA is opened."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.services import ani_bridge

router = APIRouter(prefix="/ani", tags=["ani"])


@router.post("/ensure-running")
async def ensure_running() -> dict:
    """Start Ani if it is not already open (called when the IPA UI loads)."""
    if not settings.ani_autostart:
        return {
            "ok": True,
            "skipped": True,
            "message": "ANI_AUTOSTART is disabled",
            "already_running": ani_bridge.ani_process_running(),
        }
    try:
        result = await asyncio.to_thread(ani_bridge.ensure_ani_running)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start Ani: {exc}") from exc
    return {
        "ok": True,
        "skipped": False,
        "message": result.get("message", "Ani ready"),
        "already_running": bool(result.get("already_running")),
        "project_dir": result.get("project_dir", ""),
        "pid": result.get("pid"),
    }


@router.get("/status")
async def ani_global_status() -> dict:
    root = ani_bridge.resolve_ani_root()
    return {
        "ani_autostart": settings.ani_autostart,
        "ani_root": str(root),
        "ani_version": ani_bridge.ani_version(),
        "ani_python_ok": (root / ".venv" / "Scripts" / "python.exe").is_file(),
        "independent_ok": (root / "run_ani.bat").is_file(),
        "bundled_in_repo": root == (settings.repo_root / "Ani").resolve(),
        "running": ani_bridge.ani_process_running(),
        "default_workspace": str(ani_bridge.default_ani_workspace()),
        "upgrade_hint": "Update files in Ani/, then restart Ani (or IPA). IPA always loads from Ani/.",
    }
