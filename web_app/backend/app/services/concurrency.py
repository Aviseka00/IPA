"""Shared concurrency limits for multi-user inference."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.config import settings

_inference_sem: asyncio.Semaphore | None = None


def inference_semaphore() -> asyncio.Semaphore:
    global _inference_sem
    if _inference_sem is None:
        n = max(1, int(settings.inference_max_concurrent))
        _inference_sem = asyncio.Semaphore(n)
    return _inference_sem


@asynccontextmanager
async def inference_slot():
    """Only N heavy model runs at once across the process (protects RAM/GPU)."""
    sem = inference_semaphore()
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()
