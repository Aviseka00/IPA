"""Reset a user's password. Usage: python scripts/reset_password.py EMPLOYEE_ID NEW_PASSWORD"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_db
from app.security import hash_password


async def main(employee_id: str, password: str) -> None:
    db = get_db()
    result = await db.users.update_one(
        {"employee_id": employee_id},
        {"$set": {"password_hash": hash_password(password)}},
    )
    if result.matched_count == 0:
        print(f"No user with employee_id={employee_id!r}")
        sys.exit(1)
    print(f"Password updated for {employee_id}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
