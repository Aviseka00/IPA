"""Pydantic API models."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AuthRegister(BaseModel):
    employee_id: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)


class AuthLogin(BaseModel):
    employee_id: str
    password: str


class AuthResetPassword(BaseModel):
    employee_id: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    reset_code: str = Field(min_length=4, max_length=64)
    new_password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    employee_id: str
    role: str = "operator"


class UserOut(BaseModel):
    employee_id: str
    created_at: datetime
    role: str = "operator"


class ElectronicSignatureRequest(BaseModel):
    """21 CFR Part 11 §11.50 / §11.70 electronic signature for publish."""
    password: str = Field(min_length=1, max_length=128)
    meaning: str = Field(
        default="I have reviewed this AI analysis result and approve the human-verified count as accurate.",
        min_length=10,
        max_length=500,
    )
    printed_name: str = Field(min_length=2, max_length=128)


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool


class ReviewMarker(BaseModel):
    id: str
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    type: Literal["false_negative", "false_positive"]
    note: str = ""


class AiDetection(BaseModel):
    id: str
    x: float
    y: float
    class_id: int
    confidence: float
    box: list[float] | None = None


class ReviewStateOut(BaseModel):
    item_id: str
    review_status: Literal["draft", "published"] = "draft"
    ai_infected_count: int
    ai_artifact_count: int
    human_infected_count: int
    false_negative_count: int
    false_positive_count: int
    markers: list[ReviewMarker] = []
    ai_detections: list[AiDetection] = []
    published_at: datetime | None = None
    published_by: str | None = None
    electronic_signature: dict | None = None


class ReviewSaveRequest(BaseModel):
    markers: list[ReviewMarker] = []


class PublishRequest(BaseModel):
    """Publish with optional Part 11 electronic signature."""
    signature: ElectronicSignatureRequest | None = None


class AuditEntryOut(BaseModel):
    log_id: str
    item_id: str
    batch_id: str
    employee_id: str
    action: str
    detail: dict
    created_at: datetime


class AuditReportOut(BaseModel):
    batch_id: str
    model_name: str
    review_status: str
    ai_total_infected: int
    human_total_infected: int | None
    items: list[dict]
    audit_log: list[AuditEntryOut]


class AnalysisItemOut(BaseModel):
    id: str
    filename: str
    infected_count: int
    artifact_count: int
    processing_ms: float
    input_url: str
    output_url: str
    collage_url: str
    status: str
    error_message: str | None = None
    review_status: Literal["draft", "published"] = "draft"
    ai_infected_count: int = 0
    human_infected_count: int | None = None
    false_negative_count: int = 0
    false_positive_count: int = 0
    review: ReviewStateOut | None = None
    warnings: list[str] = []
    restitch_status: str | None = None
    restitch_used: bool = False


class BatchOut(BaseModel):
    id: str
    employee_id: str
    model_id: str
    model_name: str
    created_at: datetime
    status: str
    image_count: int
    total_infected: int
    ai_total_infected: int = 0
    human_total_infected: int | None = None
    review_status: Literal["draft", "partial", "published"] = "draft"
    items: list[AnalysisItemOut] = []


class BatchSummary(BaseModel):
    id: str
    model_id: str
    model_name: str
    created_at: datetime
    status: str
    image_count: int
    total_infected: int
    ai_total_infected: int = 0
    human_total_infected: int | None = None
    review_status: Literal["draft", "partial", "published"] = "draft"


class HistoryResponse(BaseModel):
    group_by: Literal["batch", "date"]
    batches: list[BatchSummary]
    by_date: dict[str, list[BatchSummary]] | None = None


class AniStatusOut(BaseModel):
    project_exists: bool
    project_dir: str = ""
    image_count: int = 0
    labeled_count: int = 0
    labeled_stems: list[str] = []
    ani_root: str = ""
    ani_python_ok: bool = False
    retrain_dataset_dir: str = ""
    last_import_at: str | None = None


class AniOpenOut(BaseModel):
    project_dir: str
    image_count: int
    pid: int | None = None
    message: str = ""


class AniImportItemOut(BaseModel):
    item_id: str
    stem: str
    human_infected_count: int
    false_negative_count: int
    false_positive_count: int


class AniImportOut(BaseModel):
    imported: int
    items: list[AniImportItemOut] = []
    retrain_dataset_dir: str = ""
    imported_at: str = ""
    message: str = ""
