"""Create or update an app user in MongoDB.

Usage:
    python scripts/ensure_user.py EMPLOYEE_ID PASSWORD
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database import get_db
from app.security import hash_password


async def main(employee_id: str, password: str) -> None:
    db = get_db()
    now = datetime.now(timezone.utc)
    password_hash = hash_password(password)

    existing = await db.users.find_one({"employee_id": employee_id})
    if existing:
        await db.users.update_one(
            {"employee_id": employee_id},
            {"$set": {"password_hash": password_hash}},
        )
        action = "updated"
    else:
        await db.users.insert_one(
            {
                "employee_id": employee_id,
                "password_hash": password_hash,
                "created_at": now,
            }
        )
        action = "created"

    await db.users.create_index("employee_id", unique=True)
    count = await db.users.count_documents({})

    print(f"User {action}: employee_id={employee_id!r}")
    print(f"Database: {settings.mongodb_db}")
    print(f"Collection: users")
    print(f"Total users in collection: {count}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    asyncio.run(main(sys.argv[1].strip(), sys.argv[2]))
