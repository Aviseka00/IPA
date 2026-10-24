"""Upload and run analysis batches."""
from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import ALLOWED_EXTENSIONS, settings
from app.database import get_db
from app.schemas import AnalysisItemOut, BatchOut, BatchSummary, HistoryResponse
from app.security import get_current_user
from app.services.analysis_helpers import batch_summary_fields, file_url as _file_url, item_to_out
from app.services.concurrency import inference_slot
from app.services.inference_runner import run_model
from app.services.model_registry import get_model, model_available
from app.services.review_service import write_audit
from app.services.storage import ensure_batch_dirs, new_batch_id

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Keep background inference tasks alive until they finish.
_running_batches: set[asyncio.Task] = set()


def _batch_summary(doc: dict, model_name: str) -> BatchSummary:
    extra = batch_summary_fields(doc)
    return BatchSummary(
        id=doc["batch_id"],
        model_id=doc["model_id"],
        model_name=model_name,
        created_at=doc["created_at"],
        status=doc["status"],
        image_count=doc.get("image_count", 0),
        total_infected=extra["total_infected"],
        ai_total_infected=extra["ai_total_infected"],
        human_total_infected=extra["human_total_infected"],
        review_status=extra["review_status"],
    )


@router.post("/run", response_model=BatchOut)
async def run_analysis(
    model_id: str = Form(...),
    restitch_mosaic: bool = Form(True),
    user: dict = Depends(get_current_user),
    files: list[UploadFile] = File(...),
) -> BatchOut:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")

    has_tiff = any(
        Path(f.filename or "").suffix.lower() in {".tif", ".tiff"} for f in files
    )
    if has_tiff:
        try:
            import tifffile  # noqa: F401
            import imagecodecs  # noqa: F401 — LZW plate scans
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "TIFF uploads require tifffile and imagecodecs. "
                    "In web_app/backend run: pip install tifffile imagecodecs"
                ),
            ) from exc
    if len(files) > settings.max_files_per_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_batch} files per batch",
        )

    try:
        spec = get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Unknown model") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if spec.kind == "fusion":
        try:
            import skimage  # noqa: F401
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="IPA4 Fusion requires scikit-image. Run: pip install scikit-image",
            )

    if not model_available(spec):
        raise HTTPException(status_code=503, detail=f"Model not available: {spec.name}")

    employee_id = user["employee_id"]
    batch_id = new_batch_id()
    dirs = ensure_batch_dirs(employee_id, batch_id)
    db = get_db()
    created = datetime.now(timezone.utc)

    await db.batches.insert_one(
        {
            "batch_id": batch_id,
            "employee_id": employee_id,
            "model_id": model_id,
            "model_name": spec.name,
            "created_at": created,
            "status": "processing",
            "image_count": 0,
            "total_infected": 0,
            "ai_total_infected": 0,
            "human_total_infected": None,
            "review_status": "draft",
            "restitch_mosaic": bool(restitch_mosaic),
        }
    )

    items_out: list[AnalysisItemOut] = []
    jobs: list[dict] = []
    skipped: list[str] = []

    for upload in files:
        suffix = Path(upload.filename or "image.jpg").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            skipped.append(upload.filename or "unknown")
            continue
        safe_name = Path(upload.filename or "image.jpg").name
        input_path = dirs["input"] / safe_name
        content = await upload.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File too large: {safe_name}")
        input_path.write_bytes(content)

        item_id = f"{batch_id}_{safe_name}"
        work = dirs["work"] / Path(safe_name).stem
        work.mkdir(parents=True, exist_ok=True)

        item_doc = {
            "item_id": item_id,
            "batch_id": batch_id,
            "employee_id": employee_id,
            "filename": safe_name,
            "model_id": model_id,
            "created_at": created,
            "status": "queued",
            "infected_count": 0,
            "artifact_count": 0,
            "processing_ms": 0.0,
            "input_path": str(input_path),
            "output_path": "",
            "collage_path": "",
            "ai_infected_count": 0,
            "ai_artifact_count": 0,
            "ai_detections": [],
            "review_markers": [],
            "review_status": "draft",
            "false_negative_count": 0,
            "false_positive_count": 0,
            "human_infected_count": None,
        }
        await db.analysis_items.insert_one(item_doc)
        items_out.append(item_to_out(item_doc, employee_id, batch_id))
        jobs.append(
            {
                "item_id": item_id,
                "safe_name": safe_name,
                "input_path": input_path,
                "work": work,
            }
        )

    if not jobs:
        await db.batches.update_one(
            {"batch_id": batch_id},
            {"$set": {"status": "failed", "image_count": 0, "total_infected": 0}},
        )
        hint = ""
        if skipped:
            hint = f" Unsupported files: {', '.join(skipped)}."
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid images processed. Use JPG, PNG, or TIF."
                + hint
                + " For plate scans (.tif), ensure backend has tifffile: pip install tifffile"
            ),
        )

    await db.batches.update_one(
        {"batch_id": batch_id},
        {"$set": {"status": "processing", "image_count": len(jobs), "total_infected": 0, "ai_total_infected": 0}},
    )

    task = asyncio.create_task(
        _process_batch_jobs(
            batch_id=batch_id,
            employee_id=employee_id,
            model_id=model_id,
            dirs=dirs,
            jobs=jobs,
            restitch_mosaic=bool(restitch_mosaic),
        )
    )
    _running_batches.add(task)
    task.add_done_callback(_running_batches.discard)

    return BatchOut(
        id=batch_id,
        employee_id=employee_id,
        model_id=model_id,
        model_name=spec.name,
        created_at=created,
        status="processing",
        image_count=len(jobs),
        total_infected=0,
        ai_total_infected=0,
        human_total_infected=None,
        review_status="draft",
        items=items_out,
    )


async def _process_batch_jobs(
    *,
    batch_id: str,
    employee_id: str,
    model_id: str,
    dirs: dict,
    jobs: list[dict],
    restitch_mosaic: bool = True,
) -> None:
    """Run images one by one and publish each result as soon as it finishes."""
    db = get_db()
    errors = 0
    total_inf = 0

    for index, job in enumerate(jobs):
        item_id = job["item_id"]
        safe_name = job["safe_name"]
        input_path: Path = job["input_path"]
        work: Path = job["work"]
        await db.analysis_items.update_one({"item_id": item_id}, {"$set": {"status": "processing"}})
        try:
            async with inference_slot():
                result = await asyncio.to_thread(
                    run_model,
                    model_id,
                    input_path,
                    work,
                    restitch_mosaic=restitch_mosaic,
                )
            out_overlay = dirs["output"] / f"{Path(safe_name).stem}_overlay.jpg"
            out_collage = dirs["collage"] / f"{Path(safe_name).stem}_collage.jpg"
            shutil.copy2(result["overlay_path"], out_overlay)
            shutil.copy2(result["collage_path"], out_collage)
            infected = int(result["infected_count"])
            total_inf += infected
            restitch = result.get("restitch") or {}
            warnings = list(result.get("warnings") or [])
            # Persist restitch report next to collage for audit / download
            report_src = work / "mosaic_restitch_report.json"
            if report_src.is_file():
                shutil.copy2(report_src, dirs["output"] / f"{Path(safe_name).stem}_restitch_report.json")
            await db.analysis_items.update_one(
                {"item_id": item_id},
                {
                    "$set": {
                        "status": "completed",
                        "infected_count": infected,
                        "artifact_count": result["artifact_count"],
                        "ai_infected_count": infected,
                        "ai_artifact_count": result["artifact_count"],
                        "ai_detections": result.get("ai_detections", []),
                        "model_meta": result.get("model_meta", {}),
                        "processing_ms": result["processing_ms"],
                        "output_path": str(out_overlay),
                        "collage_path": str(out_collage),
                        "restitch": restitch,
                        "warnings": warnings,
                    }
                },
            )
            await write_audit(
                db,
                item_id=item_id,
                batch_id=batch_id,
                employee_id=employee_id,
                action="inference_complete",
                detail={
                    "model_id": model_id,
                    "ai_infected_count": infected,
                    "ai_artifact_count": result["artifact_count"],
                    "detections": len(result.get("ai_detections", [])),
                    "model_meta": result.get("model_meta", {}),
                    "restitch_status": restitch.get("status"),
                    "restitch_used": restitch.get("used_restitched"),
                    "warnings": warnings,
                    "index": index + 1,
                    "of": len(jobs),
                },
            )
        except Exception as exc:
            errors += 1
            await db.analysis_items.update_one(
                {"item_id": item_id},
                {"$set": {"status": "failed", "error": str(exc).strip() or type(exc).__name__}},
            )

        done = index + 1
        still_running = done < len(jobs)
        if still_running:
            batch_status = "processing"
        elif errors == 0:
            batch_status = "completed"
        elif errors < len(jobs):
            batch_status = "partial"
        else:
            batch_status = "failed"
        await db.batches.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "status": batch_status,
                    "image_count": len(jobs),
                    "total_infected": total_inf,
                    "ai_total_infected": total_inf,
                    "completed_count": done - errors,
                }
            },
        )


@router.get("/history", response_model=HistoryResponse)
async def history(
    group_by: Literal["batch", "date"] = "batch",
    user: dict = Depends(get_current_user),
) -> HistoryResponse:
    db = get_db()
    employee_id = user["employee_id"]
    cursor = db.batches.find({"employee_id": employee_id}).sort("created_at", -1).limit(200)
    batches: list[BatchSummary] = []
    by_date: dict[str, list[BatchSummary]] = {}

    async for doc in cursor:
        s = _batch_summary(doc, doc.get("model_name", doc["model_id"]))
        batches.append(s)
        if group_by == "date":
            created = doc.get("created_at")
            if created is None:
                continue
            key = created.strftime("%Y-%m-%d")
            by_date.setdefault(key, []).append(s)

    return HistoryResponse(
        group_by=group_by,
        batches=batches,
        by_date=by_date if group_by == "date" else None,
    )


@router.get("/batches/{batch_id}", response_model=BatchOut)
async def get_batch(
    batch_id: str,
    user: dict = Depends(get_current_user),
) -> BatchOut:
    db = get_db()
    doc = await db.batches.find_one(
        {"batch_id": batch_id, "employee_id": user["employee_id"]}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Batch not found")

    items: list[AnalysisItemOut] = []
    async for item in db.analysis_items.find({"batch_id": batch_id}):
        items.append(item_to_out(item, user["employee_id"], batch_id))

    extra = batch_summary_fields(doc)
    return BatchOut(
        id=batch_id,
        employee_id=doc["employee_id"],
        model_id=doc["model_id"],
        model_name=doc.get("model_name", doc["model_id"]),
        created_at=doc["created_at"],
        status=doc["status"],
        image_count=doc.get("image_count", len(items)),
        total_infected=extra["total_infected"],
        ai_total_infected=extra["ai_total_infected"],
        human_total_infected=extra["human_total_infected"],
        review_status=extra["review_status"],
        items=items,
    )
