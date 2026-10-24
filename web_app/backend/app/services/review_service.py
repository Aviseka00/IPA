"""Human review helpers — markers, counts, audit, reviewed overlay."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

MarkerType = Literal["false_negative", "false_positive"]


def new_marker_id() -> str:
    return uuid.uuid4().hex[:12]


def compute_human_infected(ai_count: int, markers: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return (human_count, fn_count, fp_count)."""
    fn = sum(1 for m in markers if m.get("type") == "false_negative")
    fp = sum(1 for m in markers if m.get("type") == "false_positive")
    human = max(0, ai_count + fn - fp)
    return human, fn, fp


async def write_audit(
    db: Any,
    *,
    item_id: str,
    batch_id: str,
    employee_id: str,
    action: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Part 11 hash-chained audit (delegates to system audit + review mirror)."""
    from app.services.compliance_controls import write_system_audit

    await write_system_audit(
        db,
        employee_id=employee_id,
        action=action,
        detail=detail,
        item_id=item_id,
        batch_id=batch_id,
        record_type="review",
    )


def draw_review_overlay(
    base_bgr: np.ndarray,
    markers: list[dict[str, Any]],
    ai_detections: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Draw AI boxes (faint), green FN dots, red FP dots."""
    out = base_bgr.copy()
    h, w = out.shape[:2]

    for det in ai_detections or []:
        if det.get("class_id", 0) != 0:
            continue
        box = det.get("box")
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(v * dim) for v, dim in zip(box, (w, h, w, h))]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 80), 1)

    for m in markers:
        x = int(float(m["x"]) * w)
        y = int(float(m["y"]) * h)
        if m.get("type") == "false_negative":
            color = (0, 220, 80)  # BGR green
            label = "FN"
        else:
            color = (60, 60, 255)  # BGR red
            label = "FP"
        cv2.circle(out, (x, y), 10, color, -1)
        cv2.circle(out, (x, y), 12, (255, 255, 255), 2)
        cv2.putText(out, label, (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return out


def normalize_markers(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for m in raw:
        t = m.get("type")
        if t not in ("false_negative", "false_positive"):
            continue
        cleaned.append(
            {
                "id": m.get("id") or new_marker_id(),
                "x": max(0.0, min(1.0, float(m["x"]))),
                "y": max(0.0, min(1.0, float(m["y"]))),
                "type": t,
                "note": str(m.get("note") or ""),
            }
        )
    return cleaned
