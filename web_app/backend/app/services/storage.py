"""Per-user filesystem layout."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import settings


def user_root(employee_id: str) -> Path:
    return settings.storage_root / "users" / employee_id


def batch_dir(employee_id: str, batch_id: str) -> Path:
    return user_root(employee_id) / "batches" / batch_id


def new_batch_id() -> str:
    return uuid4().hex[:16]


def ensure_batch_dirs(employee_id: str, batch_id: str) -> dict[str, Path]:
    base = batch_dir(employee_id, batch_id)
    dirs = {
        "base": base,
        "input": base / "input",
        "output": base / "output",
        "collage": base / "collage",
        "work": base / "work",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
