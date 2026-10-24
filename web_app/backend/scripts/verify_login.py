"""Verify login credentials against MongoDB. Usage: python scripts/verify_login.py EMPLOYEE_ID PASSWORD"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_db
from app.security import verify_password


async def main(employee_id: str, password: str) -> None:
    db = get_db()
    user = await db.users.find_one({"employee_id": employee_id})
    if not user:
        all_ids = await db.users.find({}, {"employee_id": 1}).to_list(50)
        print(f"NOT FOUND: employee_id={employee_id!r}")
        print("Registered IDs:", [u.get("employee_id") for u in all_ids])
        sys.exit(1)
    ok = verify_password(password, user.get("password_hash", ""))
    print(f"employee_id={employee_id!r} verify={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
