"""MongoDB client and job persistence."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from .config import MONGODB_DB, MONGODB_URI

_client: MongoClient | None = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client[MONGODB_DB]


def jobs_collection() -> Collection:
    return get_db()["analysis_jobs"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def insert_job(doc: dict[str, Any]) -> str:
    jobs_collection().insert_one(doc)
    return doc["job_id"]


def find_job(job_id: str) -> dict[str, Any] | None:
    return jobs_collection().find_one({"job_id": job_id})
