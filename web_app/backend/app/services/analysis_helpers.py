"""Shared helpers for analysis item serialization."""
from __future__ import annotations

from pathlib import Path

from app.schemas import AnalysisItemOut, ReviewMarker, ReviewStateOut


def file_url(employee_id: str, batch_id: str, folder: str, filename: str) -> str:
    return f"/api/files/{employee_id}/{batch_id}/{folder}/{filename}"


def item_to_out(item: dict, employee_id: str, batch_id: str) -> AnalysisItemOut:
    fn = item["filename"]
    stem = Path(fn).stem
    completed = item.get("status") == "completed"
    markers = item.get("review_markers") or []
    ai_inf = item.get("ai_infected_count", item.get("infected_count", 0))
    review = ReviewStateOut(
        item_id=item["item_id"],
        review_status=item.get("review_status", "draft"),
        ai_infected_count=ai_inf,
        ai_artifact_count=item.get("ai_artifact_count", item.get("artifact_count", 0)),
        human_infected_count=item.get("human_infected_count")
        if item.get("human_infected_count") is not None
        else max(0, ai_inf + sum(1 for m in markers if m.get("type") == "false_negative") - sum(1 for m in markers if m.get("type") == "false_positive")),
        false_negative_count=item.get("false_negative_count", sum(1 for m in markers if m.get("type") == "false_negative")),
        false_positive_count=item.get("false_positive_count", sum(1 for m in markers if m.get("type") == "false_positive")),
        markers=[ReviewMarker(**m) for m in markers],
        ai_detections=item.get("ai_detections") or [],
        published_at=item.get("published_at"),
        published_by=item.get("published_by"),
    )
    display_count = (
        item.get("human_infected_count")
        if item.get("human_infected_count") is not None
        and (
            item.get("review_status") == "published"
            or item.get("label_source") == "ani_ground_truth"
        )
        else ai_inf
    )
    return AnalysisItemOut(
        id=item["item_id"],
        filename=fn,
        infected_count=display_count,
        artifact_count=item.get("artifact_count", 0),
        processing_ms=item.get("processing_ms", 0.0),
        input_url=file_url(employee_id, batch_id, "input", fn),
        output_url=file_url(employee_id, batch_id, "output", f"{stem}_overlay.jpg") if completed else "",
        collage_url=file_url(employee_id, batch_id, "collage", f"{stem}_collage.jpg") if completed else "",
        status=item.get("status", "unknown"),
        error_message=item.get("error"),
        review_status=item.get("review_status", "draft"),
        ai_infected_count=ai_inf,
        human_infected_count=item.get("human_infected_count"),
        false_negative_count=review.false_negative_count,
        false_positive_count=review.false_positive_count,
        review=review,
        warnings=list(item.get("warnings") or []),
        restitch_status=(item.get("restitch") or {}).get("status"),
        restitch_used=bool((item.get("restitch") or {}).get("used_restitched")),
    )


def batch_summary_fields(doc: dict) -> dict:
    return {
        "ai_total_infected": doc.get("ai_total_infected", doc.get("total_infected", 0)),
        "human_total_infected": doc.get("human_total_infected"),
        "review_status": doc.get("review_status", "draft"),
        "total_infected": doc.get("human_total_infected")
        if doc.get("review_status") == "published" and doc.get("human_total_infected") is not None
        else doc.get("total_infected", 0),
    }
