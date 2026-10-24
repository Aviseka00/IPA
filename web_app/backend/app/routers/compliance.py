"""21 CFR Part 11 / GAMP 5 compliance status and controls API."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.config import settings
from app.database import get_db
from app.security import get_current_user

router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.get("/status")
async def compliance_status(user: dict = Depends(get_current_user)) -> dict:
    """Machine-readable Part 11 / GAMP 5 control matrix for CSV evidence."""
    db = get_db()
    audit_count = await db.system_audit.count_documents({})
    review_audit_count = await db.review_audit.count_documents({})
    user_count = await db.users.count_documents({})

    controls = [
        {
            "id": "11.10.a",
            "title": "System validation",
            "status": "documented",
            "evidence": "web_app/validation/gamp5/ — URS, Risk Assessment, IQ/OQ/PQ protocols",
        },
        {
            "id": "11.10.c",
            "title": "Authority checks / unique user ID",
            "status": "implemented",
            "evidence": "Unique employee_id, bcrypt passwords, JWT sessions, roles (operator/reviewer/qa/admin)",
        },
        {
            "id": "11.10.d",
            "title": "Device / operational checks",
            "status": "implemented",
            "evidence": "Model availability checks, file type validation, publish only after review",
        },
        {
            "id": "11.10.e",
            "title": "Audit trail",
            "status": "implemented",
            "evidence": f"Append-only hash-chained system_audit ({audit_count} entries) + review_audit ({review_audit_count})",
        },
        {
            "id": "11.10.g",
            "title": "Authority checks for e-records",
            "status": "implemented",
            "evidence": "Role field on users; publish requires authenticated user + e-signature",
        },
        {
            "id": "11.10.h",
            "title": "Device checks for data validity",
            "status": "implemented",
            "evidence": "Upload extension allow-list, size limits, TIFF codec dependency checks",
        },
        {
            "id": "11.10.i",
            "title": "Operational system checks",
            "status": "implemented",
            "evidence": "Draft → human review → e-signed publish workflow before final count",
        },
        {
            "id": "11.10.k",
            "title": "Sequence / determinism",
            "status": "implemented",
            "evidence": "UTC timestamps, hash chain (prev_hash → entry_hash)",
        },
        {
            "id": "11.50",
            "title": "Signature manifestations",
            "status": "implemented" if settings.require_esignature_on_publish else "configurable",
            "evidence": "Printed name, meaning, timestamp, user ID stored on publish",
        },
        {
            "id": "11.70",
            "title": "Signature / record linking",
            "status": "implemented",
            "evidence": "electronic_signature object bound to analysis_items item_id + audit entry",
        },
        {
            "id": "11.300",
            "title": "Electronic signature components / controls",
            "status": "implemented",
            "evidence": "Password re-authentication at publish; lockout; password complexity & history",
        },
    ]

    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "part11_mode": settings.part11_mode,
        "gamp_category": settings.gamp_category,
        "assessment_date": datetime.now(timezone.utc).isoformat(),
        "assessed_by": user["employee_id"],
        "record_retention_years": settings.record_retention_years,
        "session_timeout_minutes": settings.access_token_expire_minutes,
        "open_registration_allowed": settings.allow_open_registration,
        "esignature_required_on_publish": settings.require_esignature_on_publish,
        "password_policy": {
            "min_length": settings.password_min_length,
            "require_upper": settings.password_require_upper,
            "require_lower": settings.password_require_lower,
            "require_digit": settings.password_require_digit,
            "require_special": settings.password_require_special,
            "history_count": settings.password_history_count,
            "max_failed_logins": settings.max_failed_logins,
            "lockout_minutes": settings.lockout_minutes,
        },
        "user_count": user_count,
        "controls": controls,
        "validation_package_path": "web_app/validation/gamp5/",
        "disclaimer": (
            "Technical controls in this application support 21 CFR Part 11 and GAMP 5 CSV. "
            "Full regulatory compliance also requires executed IQ/OQ/PQ, SOPs, training records, "
            "and organizational procedural controls signed by QA."
        ),
        "compliance_posture": "technically_controlled_pending_csv_execution",
    }


@router.get("/audit")
async def system_audit_tail(
    limit: int = 100,
    user: dict = Depends(get_current_user),
) -> list[dict]:
    """Recent system audit entries (Part 11 §11.10(e))."""
    db = get_db()
    limit = max(1, min(limit, 500))
    out: list[dict] = []
    cursor = db.system_audit.find().sort("created_at", -1).limit(limit)
    async for doc in cursor:
        out.append(
            {
                "log_id": doc.get("log_id"),
                "employee_id": doc.get("employee_id"),
                "action": doc.get("action"),
                "detail": doc.get("detail") or {},
                "created_at": doc.get("created_at"),
                "entry_hash": doc.get("entry_hash"),
                "prev_hash": doc.get("prev_hash"),
                "item_id": doc.get("item_id", ""),
                "batch_id": doc.get("batch_id", ""),
            }
        )
    return out
