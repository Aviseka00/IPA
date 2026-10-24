"""Human review — markers, publish, audit trail."""
from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.config import settings
from app.schemas import (
    AniImportItemOut,
    AniImportOut,
    AniOpenOut,
    AniStatusOut,
    AuditEntryOut,
    AuditReportOut,
    PublishRequest,
    ReviewMarker,
    ReviewSaveRequest,
    ReviewStateOut,
)
from app.security import get_current_user, verify_password
from app.services import ani_bridge
from app.services.review_service import (
    compute_human_infected,
    draw_review_overlay,
    normalize_markers,
    write_audit,
)

router = APIRouter(prefix="/analysis", tags=["review"])


async def _get_item(item_id: str, employee_id: str) -> dict:
    db = get_db()
    item = await db.analysis_items.find_one({"item_id": item_id, "employee_id": employee_id})
    if not item:
        raise HTTPException(status_code=404, detail="Analysis item not found")
    return item


def _review_state(item: dict) -> ReviewStateOut:
    markers = item.get("review_markers") or []
    ai = item.get("ai_infected_count", item.get("infected_count", 0))
    human, fn, fp = compute_human_infected(ai, markers)
    # Prefer Ani ground-truth counts (full mask) over click-marker math
    if item.get("label_source") == "ani_ground_truth" and item.get("human_infected_count") is not None:
        human = int(item["human_infected_count"])
        fn = int(item.get("false_negative_count", fn))
        fp = int(item.get("false_positive_count", fp))
    elif item.get("review_status") == "published" and item.get("human_infected_count") is not None:
        human = item["human_infected_count"]
    return ReviewStateOut(
        item_id=item["item_id"],
        review_status=item.get("review_status", "draft"),
        ai_infected_count=ai,
        ai_artifact_count=item.get("ai_artifact_count", item.get("artifact_count", 0)),
        human_infected_count=human,
        false_negative_count=fn,
        false_positive_count=fp,
        markers=[ReviewMarker(**m) for m in markers],
        ai_detections=item.get("ai_detections") or [],
        published_at=item.get("published_at"),
        published_by=item.get("published_by"),
        electronic_signature=item.get("electronic_signature"),
    )


@router.get("/items/{item_id}/review", response_model=ReviewStateOut)
async def get_review(item_id: str, user: dict = Depends(get_current_user)) -> ReviewStateOut:
    item = await _get_item(item_id, user["employee_id"])
    return _review_state(item)


@router.put("/items/{item_id}/review", response_model=ReviewStateOut)
async def save_review(
    item_id: str,
    body: ReviewSaveRequest,
    user: dict = Depends(get_current_user),
) -> ReviewStateOut:
    item = await _get_item(item_id, user["employee_id"])
    if item.get("review_status") == "published":
        raise HTTPException(status_code=400, detail="Already published — cannot edit markers")

    markers = normalize_markers([m.model_dump() for m in body.markers])
    ai = item.get("ai_infected_count", item.get("infected_count", 0))
    human, fn, fp = compute_human_infected(ai, markers)

    db = get_db()
    await db.analysis_items.update_one(
        {"item_id": item_id},
        {
            "$set": {
                "review_markers": markers,
                "review_status": "draft",
                "false_negative_count": fn,
                "false_positive_count": fp,
                "human_infected_count": human,
            }
        },
    )
    await write_audit(
        db,
        item_id=item_id,
        batch_id=item["batch_id"],
        employee_id=user["employee_id"],
        action="save_draft",
        detail={"markers": len(markers), "fn": fn, "fp": fp, "human_count": human},
    )
    item = await _get_item(item_id, user["employee_id"])
    return _review_state(item)


@router.post("/items/{item_id}/publish", response_model=ReviewStateOut)
async def publish_review(
    item_id: str,
    body: PublishRequest | None = None,
    user: dict = Depends(get_current_user),
) -> ReviewStateOut:
    item = await _get_item(item_id, user["employee_id"])
    if item.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Cannot publish failed analysis")
    if item.get("review_status") == "published":
        raise HTTPException(status_code=400, detail="Already published")

    body = body or PublishRequest()
    esig_doc = None
    if settings.require_esignature_on_publish or settings.part11_mode:
        sig = body.signature
        if not sig:
            raise HTTPException(
                status_code=400,
                detail="Electronic signature required (21 CFR Part 11) — provide password, printed name, and meaning",
            )
        if not verify_password(sig.password, user["password_hash"]):
            await write_audit(
                get_db(),
                item_id=item_id,
                batch_id=item["batch_id"],
                employee_id=user["employee_id"],
                action="esignature_failed",
                detail={"reason": "invalid_password"},
            )
            raise HTTPException(status_code=401, detail="Electronic signature failed — invalid password")
        esig_doc = {
            "employee_id": user["employee_id"],
            "printed_name": sig.printed_name.strip(),
            "meaning": sig.meaning.strip(),
            "signed_at": datetime.now(timezone.utc),
            "signature_method": "password_reauthentication",
            "regulation": "21 CFR Part 11 §§11.50, 11.70, 11.300",
        }

    markers = item.get("review_markers") or []
    ai = item.get("ai_infected_count", item.get("infected_count", 0))
    human, fn, fp = compute_human_infected(ai, markers)
    if item.get("label_source") == "ani_ground_truth" and item.get("human_infected_count") is not None:
        human = int(item["human_infected_count"])
        fn = int(item.get("false_negative_count", fn))
        fp = int(item.get("false_positive_count", fp))
    now = datetime.now(timezone.utc)

    # Regenerate reviewed overlay + collage
    output_path = Path(item.get("output_path", ""))
    if output_path.is_file():
        base = cv2.imread(str(output_path))
        if base is not None:
            reviewed = draw_review_overlay(base, markers, item.get("ai_detections"))
            reviewed_path = output_path.parent / f"{output_path.stem.replace('_overlay', '')}_reviewed_overlay.jpg"
            cv2.imwrite(str(reviewed_path), reviewed)
            shutil.copy2(reviewed_path, output_path)

    db = get_db()
    update_fields: dict = {
        "review_status": "published",
        "human_infected_count": human,
        "false_negative_count": fn,
        "false_positive_count": fp,
        "infected_count": human,
        "published_at": now,
        "published_by": user["employee_id"],
    }
    if esig_doc:
        update_fields["electronic_signature"] = esig_doc
    await db.analysis_items.update_one(
        {"item_id": item_id},
        {"$set": update_fields},
    )

    # Recompute batch human totals
    batch_id = item["batch_id"]
    total_human = 0
    all_published = True
    async for it in db.analysis_items.find({"batch_id": batch_id, "status": "completed"}):
        if it.get("review_status") != "published":
            all_published = False
            total_human += it.get("ai_infected_count", it.get("infected_count", 0))
        else:
            total_human += it.get("human_infected_count") or it.get("ai_infected_count", it.get("infected_count", 0))

    await db.batches.update_one(
        {"batch_id": batch_id},
        {
            "$set": {
                "human_total_infected": total_human,
                "total_infected": total_human,
                "review_status": "published" if all_published else "partial",
            }
        },
    )

    await write_audit(
        db,
        item_id=item_id,
        batch_id=batch_id,
        employee_id=user["employee_id"],
        action="publish_esigned" if esig_doc else "publish",
        detail={
            "ai_count": ai,
            "human_count": human,
            "false_negative": fn,
            "false_positive": fp,
            "electronic_signature": {
                "printed_name": esig_doc["printed_name"],
                "meaning": esig_doc["meaning"],
                "signed_at": esig_doc["signed_at"].isoformat(),
            }
            if esig_doc
            else None,
        },
    )

    item = await _get_item(item_id, user["employee_id"])
    return _review_state(item)


@router.get("/items/{item_id}/audit", response_model=list[AuditEntryOut])
async def item_audit(item_id: str, user: dict = Depends(get_current_user)) -> list[AuditEntryOut]:
    await _get_item(item_id, user["employee_id"])
    db = get_db()
    entries: list[AuditEntryOut] = []
    cursor = db.review_audit.find({"item_id": item_id}).sort("created_at", -1).limit(200)
    async for doc in cursor:
        entries.append(
            AuditEntryOut(
                log_id=doc["log_id"],
                item_id=doc["item_id"],
                batch_id=doc["batch_id"],
                employee_id=doc["employee_id"],
                action=doc["action"],
                detail=doc.get("detail") or {},
                created_at=doc["created_at"],
            )
        )
    return entries


@router.get("/batches/{batch_id}/ani/status", response_model=AniStatusOut)
async def ani_status(batch_id: str, user: dict = Depends(get_current_user)) -> AniStatusOut:
    db = get_db()
    batch = await db.batches.find_one({"batch_id": batch_id, "employee_id": user["employee_id"]})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    status = ani_bridge.project_status(user["employee_id"], batch_id)
    return AniStatusOut(
        **status,
        last_import_at=batch.get("ani_last_import_at"),
    )


@router.post("/batches/{batch_id}/ani/open", response_model=AniOpenOut)
async def ani_open(batch_id: str, user: dict = Depends(get_current_user)) -> AniOpenOut:
    """Build Ani project from model detections and launch Ani v0.4.0."""
    db = get_db()
    batch = await db.batches.find_one({"batch_id": batch_id, "employee_id": user["employee_id"]})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    items: list[dict] = []
    async for item in db.analysis_items.find({"batch_id": batch_id, "employee_id": user["employee_id"]}):
        items.append(item)

    try:
        project_dir = await asyncio.to_thread(
            ani_bridge.build_ani_project, user["employee_id"], batch_id, items
        )
        launch = await asyncio.to_thread(ani_bridge.launch_ani, project_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open Ani: {exc}") from exc

    n_images = len(list((project_dir / "images").glob("*"))) if (project_dir / "images").is_dir() else 0
    await db.batches.update_one(
        {"batch_id": batch_id},
        {
            "$set": {
                "ani_project_dir": str(project_dir),
                "ani_opened_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )
    await write_audit(
        db,
        item_id=f"{batch_id}_ani",
        batch_id=batch_id,
        employee_id=user["employee_id"],
        action="ani_opened",
        detail={"project_dir": str(project_dir), "image_count": n_images, "pid": launch.get("pid")},
    )
    return AniOpenOut(
        project_dir=str(project_dir),
        image_count=n_images,
        pid=launch.get("pid"),
        message="Ani launched with model detections as provisional labels. Correct FN/FP, save, then Import Ani labels.",
    )


@router.post("/batches/{batch_id}/ani/import", response_model=AniImportOut)
async def ani_import(batch_id: str, user: dict = Depends(get_current_user)) -> AniImportOut:
    """Import Ani ground-truth masks (confirmed detections + human FN/FP) into IPA + retrain pool."""
    db = get_db()
    batch = await db.batches.find_one({"batch_id": batch_id, "employee_id": user["employee_id"]})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    items: list[dict] = []
    async for item in db.analysis_items.find({"batch_id": batch_id, "employee_id": user["employee_id"]}):
        items.append(item)

    try:
        result = await asyncio.to_thread(
            ani_bridge.import_ani_labels,
            user["employee_id"],
            batch_id,
            items,
            batch.get("model_id", ""),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ani import failed: {exc}") from exc

    out_items: list[AniImportItemOut] = []
    for row in result["items"]:
        await db.analysis_items.update_one(
            {"item_id": row["item_id"]},
            {
                "$set": {
                    "human_infected_count": row["human_infected_count"],
                    "false_negative_count": row["false_negative_count"],
                    "false_positive_count": row["false_positive_count"],
                    "review_status": "draft",
                    "ani_gt_mask_path": row["gt_mask_path"],
                    "ani_gt_classes_path": row["gt_classes_path"],
                    "ani_import_stats": row.get("stats") or {},
                    "ani_retrain": row.get("retrain") or {},
                    "label_source": "ani_ground_truth",
                }
            },
        )
        await write_audit(
            db,
            item_id=row["item_id"],
            batch_id=batch_id,
            employee_id=user["employee_id"],
            action="ani_imported",
            detail={
                "human_infected_count": row["human_infected_count"],
                "fn": row["false_negative_count"],
                "fp": row["false_positive_count"],
                "gt_mask_path": row["gt_mask_path"],
                "retrain": row.get("retrain") or {},
            },
        )
        out_items.append(
            AniImportItemOut(
                item_id=row["item_id"],
                stem=row["stem"],
                human_infected_count=row["human_infected_count"],
                false_negative_count=row["false_negative_count"],
                false_positive_count=row["false_positive_count"],
            )
        )

    # Refresh draft human batch total from imported items
    human_total = 0
    async for it in db.analysis_items.find({"batch_id": batch_id, "status": "completed"}):
        if it.get("human_infected_count") is not None:
            human_total += int(it["human_infected_count"])
        else:
            human_total += int(it.get("ai_infected_count", it.get("infected_count", 0)))

    imported_at = result.get("imported_at") or datetime.now(timezone.utc).isoformat()
    await db.batches.update_one(
        {"batch_id": batch_id},
        {
            "$set": {
                "human_total_infected": human_total,
                "ani_last_import_at": imported_at,
                "review_status": "partial" if batch.get("review_status") == "published" else "draft",
            }
        },
    )
    await write_audit(
        db,
        item_id=f"{batch_id}_ani",
        batch_id=batch_id,
        employee_id=user["employee_id"],
        action="retrain_dataset_updated",
        detail={
            "imported": result["imported"],
            "retrain_dataset_dir": result.get("retrain_dataset_dir"),
            "human_total_infected": human_total,
        },
    )

    return AniImportOut(
        imported=result["imported"],
        items=out_items,
        retrain_dataset_dir=result.get("retrain_dataset_dir", ""),
        imported_at=imported_at,
        message=(
            f"Imported {result['imported']} ground-truth mask(s). "
            "Publish with e-signature when ready. Retrain dataset updated."
        ),
    )


@router.get("/batches/{batch_id}/audit-report", response_model=AuditReportOut)
async def batch_audit_report(batch_id: str, user: dict = Depends(get_current_user)) -> AuditReportOut:
    db = get_db()
    batch = await db.batches.find_one({"batch_id": batch_id, "employee_id": user["employee_id"]})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    items_summary: list[dict] = []
    async for item in db.analysis_items.find({"batch_id": batch_id}):
        items_summary.append(
            {
                "item_id": item["item_id"],
                "filename": item["filename"],
                "review_status": item.get("review_status", "draft"),
                "ai_infected_count": item.get("ai_infected_count", item.get("infected_count", 0)),
                "human_infected_count": item.get("human_infected_count"),
                "false_negative_count": item.get("false_negative_count", 0),
                "false_positive_count": item.get("false_positive_count", 0),
            }
        )

    audit_entries: list[AuditEntryOut] = []
    cursor = db.review_audit.find({"batch_id": batch_id}).sort("created_at", -1).limit(500)
    async for doc in cursor:
        audit_entries.append(
            AuditEntryOut(
                log_id=doc["log_id"],
                item_id=doc["item_id"],
                batch_id=doc["batch_id"],
                employee_id=doc["employee_id"],
                action=doc["action"],
                detail=doc.get("detail") or {},
                created_at=doc["created_at"],
            )
        )

    return AuditReportOut(
        batch_id=batch_id,
        model_name=batch.get("model_name", batch["model_id"]),
        review_status=batch.get("review_status", "draft"),
        ai_total_infected=batch.get("ai_total_infected", batch.get("total_infected", 0)),
        human_total_infected=batch.get("human_total_infected"),
        items=items_summary,
        audit_log=audit_entries,
    )
