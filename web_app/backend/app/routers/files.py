"""Serve stored user images (authenticated)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.security import decode_token, get_current_user
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

async def get_current_user_optional(token: str | None = Depends(oauth2_scheme_optional)):
    if not token:
        return None
    try:
        return await get_current_user(token)
    except Exception:
        return None
from app.services.storage import batch_dir

router = APIRouter(prefix="/files", tags=["files"])

ALLOWED_FOLDERS = {"input", "output", "collage"}


def _resolve_user(employee_id: str, cred_user: dict | None, access_token: str | None) -> str:
    if cred_user is not None:
        if cred_user["employee_id"] != employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
        return cred_user["employee_id"]
    if access_token:
        if decode_token(access_token) != employee_id:
            raise HTTPException(status_code=403, detail="Access denied")
        return employee_id
    raise HTTPException(status_code=401, detail="Not authenticated")


@router.get("/{employee_id}/{batch_id}/{folder}/{filename}")
async def get_file(
    employee_id: str,
    batch_id: str,
    folder: str,
    filename: str,
    access_token: str | None = Query(None),
    user: dict | None = Depends(get_current_user_optional),
) -> FileResponse:
    _resolve_user(employee_id, user, access_token)
    if folder not in ALLOWED_FOLDERS:
        raise HTTPException(status_code=400, detail="Invalid folder")
    path = batch_dir(employee_id, batch_id) / folder / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

