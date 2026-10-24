"""21 CFR Part 11 / GAMP 5 technical controls — password policy, roles, lockout."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings


def validate_password_complexity(password: str) -> list[str]:
    """Return list of validation errors (empty = OK). Part 11-aligned password rules."""
    errors: list[str] = []
    min_len = settings.password_min_length
    if len(password) < min_len:
        errors.append(f"Password must be at least {min_len} characters")
    if settings.password_require_upper and not re.search(r"[A-Z]", password):
        errors.append("Password must include an uppercase letter")
    if settings.password_require_lower and not re.search(r"[a-z]", password):
        errors.append("Password must include a lowercase letter")
    if settings.password_require_digit and not re.search(r"\d", password):
        errors.append("Password must include a digit")
    if settings.password_require_special and not re.search(r"[^A-Za-z0-9]", password):
        errors.append("Password must include a special character")
    return errors


def password_in_history(password: str, history_hashes: list[str], verify_fn) -> bool:
    for h in history_hashes or []:
        if verify_fn(password, h):
            return True
    return False


def audit_payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def write_system_audit(
    db: Any,
    *,
    employee_id: str,
    action: str,
    detail: dict[str, Any] | None = None,
    item_id: str = "",
    batch_id: str = "",
    record_type: str = "system",
) -> dict[str, Any]:
    """
    Append-only, hash-chained audit entry for ALCOA+ / Part 11 §11.10(e).
    Entries are never updated or deleted by application code.
    """
    created_at = datetime.now(timezone.utc)
    # Previous hash for chain integrity
    prev = await db.system_audit.find_one(sort=[("created_at", -1), ("_id", -1)])
    prev_hash = (prev or {}).get("entry_hash", "GENESIS")

    body = {
        "log_id": uuid.uuid4().hex,
        "record_type": record_type,
        "item_id": item_id,
        "batch_id": batch_id,
        "employee_id": employee_id,
        "action": action,
        "detail": detail or {},
        "created_at": created_at,
        "prev_hash": prev_hash,
        "immutable": True,
        "timezone": "UTC",
        "application": settings.app_name,
        "app_version": settings.app_version,
    }
    body["entry_hash"] = audit_payload_hash(
        {
            "log_id": body["log_id"],
            "employee_id": employee_id,
            "action": action,
            "detail": body["detail"],
            "created_at": created_at.isoformat(),
            "prev_hash": prev_hash,
            "item_id": item_id,
            "batch_id": batch_id,
        }
    )
    await db.system_audit.insert_one(body)
    # Mirror review actions into review_audit for existing UI
    if item_id or batch_id:
        await db.review_audit.insert_one(
            {
                "log_id": body["log_id"],
                "item_id": item_id,
                "batch_id": batch_id,
                "employee_id": employee_id,
                "action": action,
                "detail": detail or {},
                "created_at": created_at,
                "entry_hash": body["entry_hash"],
                "prev_hash": prev_hash,
                "immutable": True,
            }
        )
    return body


def is_account_locked(user: dict[str, Any]) -> bool:
    until = user.get("locked_until")
    if not until:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return until > datetime.now(timezone.utc)


def lock_until() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=settings.lockout_minutes)


ROLES = ("operator", "reviewer", "qa", "admin")
