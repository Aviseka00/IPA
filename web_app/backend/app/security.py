"""JWT auth, password hashing, roles, and Part 11 session controls."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from app.config import settings
from app.database import get_db
from app.services.compliance_controls import is_account_locked

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except (ValueError, TypeError, UnknownHashError):
        return False


def create_access_token(employee_id: str, role: str = "operator") -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {
        "sub": employee_id,
        "role": role,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    if not token or not str(token).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — please sign in again",
        )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token — please sign in again",
            )
        return str(sub)
    except ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired — please sign in again",
        ) from exc
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token — please sign in again",
        ) from exc


async def get_current_user_optional(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict[str, Any] | None:
    if creds is None or not creds.credentials:
        return None
    employee_id = decode_token(creds.credentials)
    db = get_db()
    user = await db.users.find_one({"employee_id": employee_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account not found — register again on this database",
        )
    if user.get("disabled"):
        raise HTTPException(status_code=403, detail="Account disabled")
    if is_account_locked(user):
        raise HTTPException(status_code=403, detail="Account locked — try again later")
    return user


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict[str, Any]:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    employee_id = decode_token(creds.credentials)
    db = get_db()
    user = await db.users.find_one({"employee_id": employee_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account not found — register again on this database",
        )
    if user.get("disabled"):
        raise HTTPException(status_code=403, detail="Account disabled")
    if is_account_locked(user):
        raise HTTPException(status_code=403, detail="Account locked — try again later")
    return user


def require_roles(*roles: str):
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "operator")
        if role not in roles and "admin" not in (role,):
            if role not in roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role}' not authorized — requires one of: {', '.join(roles)}",
                )
        return user

    return _dep
