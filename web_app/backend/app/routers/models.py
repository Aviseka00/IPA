"""List inference models."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import ModelInfo
from app.services.model_registry import list_models, model_available

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def get_models() -> list[ModelInfo]:
    out: list[ModelInfo] = []
    for spec in list_models():
        out.append(
            ModelInfo(
                id=spec.id,
                name=spec.name,
                description=spec.description,
                available=model_available(spec),
            )
        )
    return out
