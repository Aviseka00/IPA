"""IPA ↔ Ani v0.4.0 bridge: provisional masks, launch Ani, import ground-truth labels."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.services.storage import batch_dir, ensure_batch_dirs

logger = logging.getLogger(__name__)

LABEL_SET = "cells"
CLASS_INFECTED = "Infected"
CLASS_NON_INFECTED = "non-infected"


def ani_project_dir(employee_id: str, batch_id: str) -> Path:
    return batch_dir(employee_id, batch_id) / "ani_project"


def resolve_ani_root() -> Path:
    """
    IPA always uses <repo>/Ani when present so:
    - Ani can be upgraded in place (replace/update Ani/) and IPA picks it up next launch
    - Ani stays independently runnable via Ani\\run_ani.bat / repo root run_ani.bat
    Falls back to ANI_ROOT only if the bundled folder is missing.
    """
    bundled = (settings.repo_root / "Ani").resolve()
    if bundled.is_dir() and (
        (bundled / "ani").is_dir() or (bundled / "pyproject.toml").is_file()
    ):
        return bundled
    configured = Path(settings.ani_root).expanduser()
    try:
        return configured.resolve()
    except OSError:
        return configured


def ani_version() -> str:
    """Read version from Ani/pyproject.toml (updates automatically when Ani is upgraded)."""
    pyproject = resolve_ani_root() / "pyproject.toml"
    if not pyproject.is_file():
        return "unknown"
    try:
        text = pyproject.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("version"):
                # version = "0.4.0"
                part = line.split("=", 1)[-1].strip().strip("\"'")
                if part:
                    return part
    except OSError:
        pass
    return "unknown"


def _ani_python() -> Path:
    if settings.ani_python and Path(settings.ani_python).is_file():
        return Path(settings.ani_python)
    candidate = resolve_ani_root() / ".venv" / "Scripts" / "python.exe"
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def _write_mask_tif(path: Path, mask: np.ndarray) -> None:
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    tifffile.imwrite(str(tmp), mask.astype(np.uint16), compression="zlib")
    tmp.replace(path)


def _read_mask_tif(path: Path) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr.astype(np.uint16)


def _image_hw(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            import tifffile

            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                return int(arr.shape[0]), int(arr.shape[1])
            return int(arr.shape[-2]), int(arr.shape[-1])
        except Exception as exc:
            raise FileNotFoundError(f"Cannot read image: {path}") from exc
    return int(img.shape[0]), int(img.shape[1])


def detections_to_mask(
    detections: list[dict[str, Any]],
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[int, str]]:
    """Rasterize infected detection boxes into an instance label mask."""
    mask = np.zeros((height, width), dtype=np.uint16)
    classes: dict[int, str] = {}
    label_id = 0
    for det in detections or []:
        if int(det.get("class_id", 0)) != 0:
            continue
        box = det.get("box")
        if not box or len(box) != 4:
            # fall back to a small square around center
            cx = float(det.get("x", 0.5)) * width
            cy = float(det.get("y", 0.5)) * height
            half = max(8, int(0.02 * min(height, width)))
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)
        else:
            x1 = int(float(box[0]) * width)
            y1 = int(float(box[1]) * height)
            x2 = int(float(box[2]) * width)
            y2 = int(float(box[3]) * height)
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label_id += 1
        mask[y1:y2, x1:x2] = label_id
        classes[label_id] = CLASS_INFECTED
    return mask, classes


def _write_classes_json(path: Path, classes: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in classes.items()}, indent=1), encoding="utf-8")


def _read_classes_json(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    return {int(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def compare_masks(pred: np.ndarray | None, gt: np.ndarray) -> dict[str, int]:
    """Approximate FN/FP via IoU matching between instance masks."""
    if pred is None or pred.size == 0 or int(pred.max()) == 0:
        added = len([i for i in np.unique(gt) if i > 0])
        return {"cells": added, "matched": 0, "added": added, "removed": 0, "modified": 0}

    pred_ids = [int(i) for i in np.unique(pred) if i > 0]
    gt_ids = [int(i) for i in np.unique(gt) if i > 0]
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pid in pred_ids:
        pbin = pred == pid
        parea = int(pbin.sum())
        if parea == 0:
            continue
        best_iou, best_gid = 0.0, None
        for gid in gt_ids:
            if gid in matched_gt:
                continue
            gbin = gt == gid
            inter = int(np.logical_and(pbin, gbin).sum())
            if inter == 0:
                continue
            union = parea + int(gbin.sum()) - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_gid is not None and best_iou >= 0.3:
            matched_pred.add(pid)
            matched_gt.add(best_gid)
    return {
        "cells": len(gt_ids),
        "matched": len(matched_gt),
        "added": len(gt_ids) - len(matched_gt),  # FN-like
        "removed": len(pred_ids) - len(matched_pred),  # FP-like
        "modified": 0,
    }


def count_infected(mask: np.ndarray, classes: dict[int, str]) -> int:
    """Count instance IDs labeled Infected (default when class missing)."""
    ids = [int(i) for i in np.unique(mask) if i > 0]
    if not classes:
        return len(ids)
    infected = 0
    for i in ids:
        name = str(classes.get(i, CLASS_INFECTED)).strip().lower()
        if name in {"non-infected", "non_infected", "background", "negative"}:
            continue
        if "non" in name and "infect" in name:
            continue
        infected += 1
    return infected


def build_ani_project(
    employee_id: str,
    batch_id: str,
    items: list[dict[str, Any]],
) -> Path:
    """Create/update Ani project from completed analysis items."""
    dirs = ensure_batch_dirs(employee_id, batch_id)
    root = ani_project_dir(employee_id, batch_id)
    for sub in ("images", "predictions", "labels", "history", "exports", ".recovery"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "labels" / LABEL_SET).mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "version": 2,
        "label_sets": {LABEL_SET: [CLASS_INFECTED, CLASS_NON_INFECTED]},
        "images": {},
        "ipa": {
            "batch_id": batch_id,
            "employee_id": employee_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    if (root / "project.json").is_file():
        try:
            existing = json.loads((root / "project.json").read_text(encoding="utf-8"))
            meta["images"] = existing.get("images") or {}
        except Exception:
            pass

    completed = [it for it in items if it.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed images in this batch to open in Ani")

    for item in completed:
        filename = item["filename"]
        stem = Path(filename).stem
        src = Path(item.get("input_path") or (dirs["input"] / filename))
        if not src.is_file():
            logger.warning("Missing input for %s — skip", filename)
            continue
        dest_img = root / "images" / filename
        if not dest_img.is_file() or dest_img.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest_img)

        h, w = _image_hw(dest_img)
        detections = item.get("ai_detections") or []
        mask, classes = detections_to_mask(detections, h, w)
        pred_path = root / "predictions" / f"{stem}.tif"
        _write_mask_tif(pred_path, mask)
        _write_classes_json(root / "predictions" / f"{stem}.classes.json", classes)

        info = meta["images"].setdefault(stem, {})
        info.setdefault("status", "todo")
        info["ipa_item_id"] = item["item_id"]
        info["ai_infected_count"] = int(item.get("ai_infected_count", item.get("infected_count", 0)))

    (root / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def default_ani_workspace() -> Path:
    """Shared Ani project opened by default when IPA starts."""
    return Path(settings.storage_root) / "ani_workspace"


def ensure_default_workspace() -> Path:
    """Create the default IPA↔Ani workspace if missing."""
    root = default_ani_workspace()
    for sub in ("images", "predictions", "labels", "history", "exports", ".recovery"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "labels" / LABEL_SET).mkdir(parents=True, exist_ok=True)
    meta_path = root / "project.json"
    if not meta_path.is_file():
        meta = {
            "version": 2,
            "label_sets": {LABEL_SET: [CLASS_INFECTED, CLASS_NON_INFECTED]},
            "images": {},
            "ipa": {"role": "default_workspace", "note": "Opened automatically with IPA"},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def ani_process_running() -> bool:
    """True if an Ani / napari-ani process appears to be running."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore

    if psutil is not None:
        for proc in psutil.process_iter(["cmdline", "name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmd = " ".join(proc.info.get("cmdline") or []).lower()
                if "ani.exe" in name:
                    return True
                if "-m ani" in cmd or "python -m ani" in cmd or "ani\\__main__" in cmd.replace("/", "\\"):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[attr-defined]
                continue
        return False

    # Fallback without psutil (Windows)
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "process", "get", "CommandLine"],
                text=True,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,  # type: ignore[attr-defined]
            ).lower()
            return "-m ani" in out or "ani.exe" in out
        except Exception:
            return False
    return False


def launch_ani(project_dir: Path | None = None, *, force: bool = False) -> dict[str, Any]:
    """Start Ani full workspace on the project folder (non-blocking).

    Always launches the Ani install resolved from this repo (or ANI_ROOT override),
    so upgrading files under Ani/ is picked up the next time Ani starts.
    """
    if project_dir is None:
        project_dir = ensure_default_workspace()
    project_dir = Path(project_dir)
    if not (project_dir / "project.json").is_file():
        raise FileNotFoundError(f"Ani project not found: {project_dir}")

    ani_root = resolve_ani_root()
    version = ani_version()

    if not force and ani_process_running():
        return {
            "project_dir": str(project_dir),
            "pid": None,
            "already_running": True,
            "command": [],
            "ani_python": str(_ani_python()),
            "ani_root": str(ani_root),
            "ani_version": version,
            "message": f"Ani {version} is already running",
        }

    py = _ani_python()
    if not (ani_root / "ani").is_dir() and not (ani_root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Ani not found at {ani_root}. Run Ani\\setup_windows.bat after clone, "
            "or set ANI_ROOT in backend .env"
        )
    if not py.is_file() or py == Path(sys.executable):
        # IPA venv is not Ani's — require Ani's own venv for independent upgrades
        venv_py = ani_root / ".venv" / "Scripts" / "python.exe"
        if not venv_py.is_file():
            raise FileNotFoundError(
                f"Ani virtualenv missing at {venv_py}. "
                "Open a terminal in Ani\\ and run setup_windows.bat (Ani stays independently runnable)."
            )
        py = venv_py

    cmd = [str(py), "-m", "ani", "--full", str(project_dir)]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ani_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=(sys.platform != "win32"),
    )
    return {
        "project_dir": str(project_dir),
        "pid": proc.pid,
        "already_running": False,
        "command": cmd,
        "ani_python": str(py),
        "ani_root": str(ani_root),
        "ani_version": version,
        "message": f"Ani {version} started from {ani_root}",
    }


def ensure_ani_running() -> dict[str, Any]:
    """Used on IPA startup / browser open — start Ani if not already up."""
    workspace = ensure_default_workspace()
    result = launch_ani(workspace)
    result["autostart"] = True
    return result


def project_status(employee_id: str, batch_id: str) -> dict[str, Any]:
    root = ani_project_dir(employee_id, batch_id)
    labels_dir = root / "labels" / LABEL_SET
    labeled = sorted(p.stem for p in labels_dir.glob("*.tif")) if labels_dir.is_dir() else []
    images = sorted(p.stem for p in (root / "images").iterdir() if p.is_file()) if (root / "images").is_dir() else []
    return {
        "project_exists": (root / "project.json").is_file(),
        "project_dir": str(root) if root.exists() else "",
        "image_count": len(images),
        "labeled_count": len(labeled),
        "labeled_stems": labeled,
        "ani_root": str(settings.ani_root),
        "ani_python_ok": _ani_python().is_file(),
        "retrain_dataset_dir": str(settings.retrain_dataset_dir),
    }


def _export_retrain(
    *,
    batch_id: str,
    model_id: str,
    stem: str,
    image_src: Path,
    mask: np.ndarray,
    classes: dict[int, str],
    infected_count: int,
    stats: dict[str, int],
) -> dict[str, str]:
    out_root = Path(settings.retrain_dataset_dir)
    img_dir = out_root / "images"
    lab_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    key = f"{batch_id}_{stem}"
    ext = image_src.suffix.lower() or ".jpg"
    dest_img = img_dir / f"{key}{ext}"
    dest_mask = lab_dir / f"{key}.tif"
    dest_cls = lab_dir / f"{key}.classes.json"
    shutil.copy2(image_src, dest_img)
    _write_mask_tif(dest_mask, mask)
    _write_classes_json(dest_cls, classes)

    manifest = out_root / "manifest.jsonl"
    row = {
        "batch_id": batch_id,
        "model_id": model_id,
        "stem": stem,
        "key": key,
        "infected_count": infected_count,
        "fn_added": stats.get("added", 0),
        "fp_removed": stats.get("removed", 0),
        "matched": stats.get("matched", 0),
        "image": str(dest_img),
        "label": str(dest_mask),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    line = json.dumps(row) + "\n"
    _append_manifest_line(manifest, line)
    return {"image": str(dest_img), "label": str(dest_mask), "key": key}


_manifest_thread_lock = None


def _get_manifest_thread_lock():
    global _manifest_thread_lock
    import threading

    if _manifest_thread_lock is None:
        _manifest_thread_lock = threading.Lock()
    return _manifest_thread_lock


def _append_manifest_line(manifest: Path, line: str) -> None:
    """Thread- and process-safe append for concurrent analyst imports."""
    lock_path = manifest.with_suffix(manifest.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_manifest_thread_lock():
        with lock_path.open("a+b") as lf:
            if sys.platform == "win32":
                import msvcrt

                lf.seek(0)
                msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    with manifest.open("a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                finally:
                    lf.seek(0)
                    msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    with manifest.open("a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def import_ani_labels(
    employee_id: str,
    batch_id: str,
    items: list[dict[str, Any]],
    model_id: str = "",
) -> dict[str, Any]:
    """
    Pull Ani labels/cells masks as ground truth.
    Final mask = accepted model detections + FN − FP (as edited in Ani).
    """
    root = ani_project_dir(employee_id, batch_id)
    labels_dir = root / "labels" / LABEL_SET
    if not labels_dir.is_dir():
        raise FileNotFoundError("No Ani labels found — save corrections in Ani first")

    results: list[dict[str, Any]] = []
    for item in items:
        if item.get("status") != "completed":
            continue
        if item.get("review_status") == "published":
            continue
        stem = Path(item["filename"]).stem
        label_path = labels_dir / f"{stem}.tif"
        if not label_path.is_file():
            continue

        mask = _read_mask_tif(label_path)
        classes = _read_classes_json(labels_dir / f"{stem}.classes.json")
        # If Ani did not assign classes, treat all instances as Infected
        if not classes:
            classes = {int(i): CLASS_INFECTED for i in np.unique(mask) if i > 0}

        infected = count_infected(mask, classes)

        pred_path = root / "predictions" / f"{stem}.tif"
        pred = _read_mask_tif(pred_path) if pred_path.is_file() else None
        stats = compare_masks(pred, mask)
        fn = int(stats.get("added", 0))
        fp = int(stats.get("removed", 0))

        # Persist GT under batch output for IPA
        gt_dir = batch_dir(employee_id, batch_id) / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_mask = gt_dir / f"{stem}_gt.tif"
        gt_cls = gt_dir / f"{stem}_gt.classes.json"
        _write_mask_tif(gt_mask, mask)
        _write_classes_json(gt_cls, classes)

        image_src = root / "images" / item["filename"]
        if not image_src.is_file():
            image_src = Path(item.get("input_path") or "")
        export_paths = {}
        if image_src.is_file():
            export_paths = _export_retrain(
                batch_id=batch_id,
                model_id=model_id or item.get("model_id", ""),
                stem=stem,
                image_src=image_src,
                mask=mask,
                classes=classes,
                infected_count=infected,
                stats=stats,
            )

        results.append(
            {
                "item_id": item["item_id"],
                "stem": stem,
                "human_infected_count": infected,
                "false_negative_count": fn,
                "false_positive_count": fp,
                "gt_mask_path": str(gt_mask),
                "gt_classes_path": str(gt_cls),
                "retrain": export_paths,
                "stats": stats,
            }
        )

    if not results:
        raise FileNotFoundError(
            "No Ani label files matched this batch. In Ani, correct and save each image (labels/cells/*.tif)."
        )

    return {
        "imported": len(results),
        "items": results,
        "retrain_dataset_dir": str(settings.retrain_dataset_dir),
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
