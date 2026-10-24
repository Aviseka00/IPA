"""Registration, login, logout — 21 CFR Part 11 access controls."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.config import settings
from app.database import get_db
from app.schemas import (
    AuthLogin,
    AuthRegister,
    AuthResetPassword,
    TokenResponse,
    UserOut,
)
from app.security import create_access_token, get_current_user, hash_password, verify_password
from app.services.compliance_controls import (
    is_account_locked,
    lock_until,
    password_in_history,
    validate_password_complexity,
    write_system_audit,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request | None) -> str:
    if request is None:
        return ""
    return request.client.host if request.client else ""


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: AuthRegister, request: Request) -> UserOut:
    if settings.part11_mode and not settings.allow_open_registration:
        db = get_db()
        user_count = await db.users.count_documents({})
        if user_count > 0:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Open registration disabled (21 CFR Part 11). "
                    "Ask an administrator to create your account."
                ),
            )

    errors = validate_password_complexity(body.password)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    db = get_db()
    existing = await db.users.find_one({"employee_id": body.employee_id})
    if existing:
        raise HTTPException(status_code=400, detail="Employee ID already registered")

    user_count = await db.users.count_documents({})
    role = "admin" if user_count == 0 else "operator"
    now = datetime.now(timezone.utc)
    pwd_hash = hash_password(body.password)
    doc = {
        "employee_id": body.employee_id,
        "password_hash": pwd_hash,
        "password_history": [pwd_hash],
        "role": role,
        "created_at": now,
        "failed_login_attempts": 0,
        "locked_until": None,
        "disabled": False,
        "must_change_password": False,
        "last_password_change": now,
    }
    try:
        await db.users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=400, detail="Employee ID already registered"
        ) from None
    except PyMongoError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable — check MongoDB connection",
        ) from exc

    await write_system_audit(
        db,
        employee_id=body.employee_id,
        action="user_created",
        detail={"role": role, "ip": _client_ip(request)},
    )
    return UserOut(
        employee_id=body.employee_id,
        created_at=doc["created_at"],
        role=role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: AuthLogin, request: Request) -> TokenResponse:
    employee_id = body.employee_id.strip()
    password = body.password.strip()
    db = get_db()
    user = await db.users.find_one({"employee_id": employee_id})

    if user and user.get("disabled"):
        await write_system_audit(
            db, employee_id=employee_id, action="login_denied_disabled", detail={"ip": _client_ip(request)}
        )
        raise HTTPException(status_code=403, detail="Account disabled — contact administrator")

    if user and is_account_locked(user):
        await write_system_audit(
            db, employee_id=employee_id, action="login_denied_locked", detail={"ip": _client_ip(request)}
        )
        raise HTTPException(
            status_code=403,
            detail=f"Account locked after failed attempts — wait {settings.lockout_minutes} minutes",
        )

    if not user or not verify_password(password, user["password_hash"]):
        if user:
            attempts = int(user.get("failed_login_attempts", 0)) + 1
            update: dict = {"failed_login_attempts": attempts}
            if attempts >= settings.max_failed_logins:
                update["locked_until"] = lock_until()
            await db.users.update_one({"employee_id": employee_id}, {"$set": update})
            await write_system_audit(
                db,
                employee_id=employee_id,
                action="login_failed",
                detail={"attempts": attempts, "ip": _client_ip(request)},
            )
        else:
            await write_system_audit(
                db,
                employee_id=employee_id or "UNKNOWN",
                action="login_failed_unknown_user",
                detail={"ip": _client_ip(request)},
            )
        raise HTTPException(status_code=401, detail="Invalid employee ID or password")

    await db.users.update_one(
        {"employee_id": employee_id},
        {"$set": {"failed_login_attempts": 0, "locked_until": None, "last_login_at": datetime.now(timezone.utc)}},
    )
    role = user.get("role", "operator")
    token = create_access_token(employee_id, role=role)
    await write_system_audit(
        db,
        employee_id=employee_id,
        action="login_success",
        detail={"role": role, "ip": _client_ip(request)},
    )
    return TokenResponse(access_token=token, employee_id=employee_id, role=role)


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)) -> UserOut:
    return UserOut(
        employee_id=user["employee_id"],
        created_at=user["created_at"],
        role=user.get("role", "operator"),
    )


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)) -> dict[str, str]:
    db = get_db()
    await write_system_audit(
        db, employee_id=user["employee_id"], action="logout", detail={}
    )
    return {"message": "Signed out — discard token on client"}


@router.post("/reset-password")
async def reset_password(body: AuthResetPassword, request: Request) -> dict[str, str]:
    """Admin-assisted password reset using PASSWORD_RESET_TOKEN from server .env."""
    token = settings.password_reset_token.strip()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="Password reset is disabled — contact your administrator",
        )
    if body.reset_code != token:
        await write_system_audit(
            db=get_db(),
            employee_id=body.employee_id,
            action="password_reset_denied",
            detail={"ip": _client_ip(request)},
        )
        raise HTTPException(status_code=403, detail="Invalid reset code")

    errors = validate_password_complexity(body.new_password)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    db = get_db()
    user = await db.users.find_one({"employee_id": body.employee_id})
    if not user:
        raise HTTPException(status_code=404, detail="Employee ID not found")

    history = list(user.get("password_history") or [])
    if password_in_history(body.new_password, history, verify_password):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reuse any of the last {settings.password_history_count} passwords",
        )

    new_hash = hash_password(body.new_password)
    history = ([new_hash] + history)[: settings.password_history_count]
    await db.users.update_one(
        {"employee_id": body.employee_id},
        {
            "$set": {
                "password_hash": new_hash,
                "password_history": history,
                "failed_login_attempts": 0,
                "locked_until": None,
                "last_password_change": datetime.now(timezone.utc),
            }
        },
    )
    await write_system_audit(
        db,
        employee_id=body.employee_id,
        action="password_reset",
        detail={"ip": _client_ip(request)},
    )
    return {"message": f"Password updated for {body.employee_id}"}
