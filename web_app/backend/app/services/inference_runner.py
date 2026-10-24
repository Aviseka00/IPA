"""Run IPA models and build side-by-side collages."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.model_registry import ModelSpec, get_model

REPO = settings.repo_root
IPA4_DIR = REPO / "ipa4"
ML_DIR = REPO / "ml_infection"
PY = sys.executable

if str(IPA4_DIR) not in sys.path:
    sys.path.insert(0, str(IPA4_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _apply_mosaic_restitch(image_path: Path, work_dir: Path, enabled: bool) -> tuple[Path, dict]:
    """
    Standalone EVOS mosaic preprocessing before any model call.

    Returns (path_for_detection, restitch_meta). Never modifies model code.
    """
    meta: dict = {
        "enabled": bool(enabled),
        "status": "skipped",
        "warnings": [],
        "report": {},
        "used_restitched": False,
    }
    if not enabled:
        return image_path, meta
    try:
        from preprocessing.mosaic_restitch import RestitchConfig, restitch_if_needed
        from preprocessing.mosaic_restitch.io import save_image

        cfg = RestitchConfig(
            low_confidence_policy=settings.mosaic_restitch_on_low_confidence,
        )
        result = restitch_if_needed(image_path, cfg)
        meta["status"] = result.status
        meta["report"] = result.report
        meta["warnings"] = list(result.report.get("warnings") or [])
        report_path = work_dir / "mosaic_restitch_report.json"
        report_path.write_text(
            json.dumps({"status": result.status, "report": result.report}, indent=2),
            encoding="utf-8",
        )
        meta["report_path"] = str(report_path)

        use_restitched = result.status == "restitched_ok"
        if result.status == "restitched_low_confidence" and settings.mosaic_restitch_on_low_confidence == "use_restitched":
            use_restitched = True
        if result.status in ("restitched_low_confidence", "failed"):
            meta["warnings"].append(
                f"Mosaic restitch {result.status}: detection used "
                f"{'restitched' if use_restitched else 'original'} image."
            )
        if result.report.get("low_resolution"):
            warn = (
                "Input is the instrument's downscaled tiled preview; detection accuracy "
                "is reduced. Upload single-field TIFFs for analysis."
            )
            if warn not in meta["warnings"]:
                meta["warnings"].append(warn)

        if use_restitched:
            out = work_dir / "restitched.tif"
            save_image(out, result.image)
            meta["used_restitched"] = True
            meta["restitched_path"] = str(out)
            return out, meta
        return image_path, meta
    except Exception as exc:
        meta["status"] = "failed"
        meta["warnings"] = [f"restitch_hook_error:{exc}"]
        return image_path, meta


def _resize_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h == target_h:
        return bgr
    scale = target_h / h
    return cv2.resize(bgr, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def make_collage(
    raw_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    *,
    left_title: str = "Original",
    right_title: str = "Detection",
    gap: int = 8,
    banner_h: int = 36,
) -> np.ndarray:
    target_h = max(raw_bgr.shape[0], overlay_bgr.shape[0])
    body = np.hstack(
        [
            _resize_to_height(raw_bgr, target_h),
            np.full((target_h, gap, 3), 240, dtype=np.uint8),
            _resize_to_height(overlay_bgr, target_h),
        ]
    )
    w = body.shape[1]
    rw = _resize_to_height(raw_bgr, target_h).shape[1]
    banner = np.full((banner_h, w, 3), 32, dtype=np.uint8)
    cv2.putText(banner, left_title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        banner, right_title, (rw + gap + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 220, 255), 2
    )
    return np.vstack([banner, body])


def _load_image(path: Path) -> np.ndarray:
    from utils import load_image

    return load_image(path)


def run_yolo(spec: ModelSpec, image_path: Path, conf: float, device: str) -> tuple[np.ndarray, int, int, list[dict]]:
    from ultralytics import YOLO

    bgr = _load_image(image_path)
    h, w = bgr.shape[:2]
    res = YOLO(str(spec.weights)).predict(
        source=bgr, imgsz=spec.imgsz, device=device, conf=conf, verbose=False
    )[0]
    plotted = res.plot()
    if plotted is None:
        plotted = bgr.copy()
    n_inf, n_art = 0, 0
    detections: list[dict] = []
    if res.boxes is not None and len(res.boxes):
        cls = res.boxes.cls.cpu().numpy().astype(int)
        n_inf = int((cls == 0).sum())
        n_art = int((cls == 1).sum())
        for i, box in enumerate(res.boxes):
            xyxy = box.xyxy.cpu().numpy()[0]
            cls_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            detections.append(
                {
                    "id": f"det_{i}",
                    "x": float((xyxy[0] + xyxy[2]) / 2 / w),
                    "y": float((xyxy[1] + xyxy[3]) / 2 / h),
                    "class_id": cls_id,
                    "confidence": confidence,
                    "box": [
                        float(xyxy[0] / w),
                        float(xyxy[1] / h),
                        float(xyxy[2] / w),
                        float(xyxy[3] / h),
                    ],
                }
            )
    return plotted, n_inf, n_art, detections


def run_fusion(spec: ModelSpec, image_path: Path, conf: float, tmp: Path) -> tuple[np.ndarray, int, int, list[dict]]:
    from ipa4_model import IPA4Detector

    det = IPA4Detector(spec.weights, config_path=settings.ipa4_config)
    res = det.detect(image_path, conf=conf)
    raw = _load_image(image_path)
    h, w = raw.shape[:2]
    count = det.visualize_fusion_qc(res, tmp, bgr=raw)
    overlay = cv2.imread(str(tmp))
    if overlay is None:
        overlay = raw.copy()
    detections: list[dict] = []
    for i, cell in enumerate(res.detections):
        if cell.class_id != 0:
            continue
        x1, y1, x2, y2 = cell.box_xyxy
        detections.append(
            {
                "id": f"det_{i}",
                "x": float((x1 + x2) / 2 / w),
                "y": float((y1 + y2) / 2 / h),
                "class_id": cell.class_id,
                "confidence": float(cell.confidence),
                "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
            }
        )
    return overlay, count, int(res.artifact_count), detections


def run_ipa6(spec: ModelSpec, image_path: Path, device: str) -> tuple[np.ndarray, int, int, list[dict]]:
    """Tiled native-resolution IPA6 predictor (ipa4/ipa6_infer.py)."""
    from ipa6_infer import IPA6Config, IPA6Predictor

    cfg = IPA6Config.load(spec.weights.with_name("ipa6_config.json") if spec.weights else None)
    cfg.device = "cpu" if not device or device == "cpu" else device
    predictor = IPA6Predictor(spec.weights, config=cfg)
    bgr = _load_image(image_path)
    h, w = bgr.shape[:2]
    det = predictor.predict(bgr)
    overlay = bgr.copy()
    detections: list[dict] = []
    boxes = det["boxes"]
    confs = det["conf"]
    for i, (box, score) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        detections.append(
            {
                "id": f"det_{i}",
                "x": float(((x1 + x2) / 2) / w),
                "y": float(((y1 + y2) / 2) / h),
                "class_id": 0,
                "confidence": float(score),
                "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
            }
        )
    n_inf = int(det["count"])
    cv2.putText(
        overlay,
        f"IPA6 infected={n_inf}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return overlay, n_inf, 0, detections


_ipa8_single_cache: dict[str, object] = {}


def run_ipa8_single(spec: ModelSpec, image_path: Path, device: str) -> tuple[np.ndarray, int, int, list[dict]]:
    """One IPA8 weight (A or B) with the same tiles, without the second model voting."""
    from ipa6_infer import IPA6Config, IPA6Predictor

    dev = "" if not device or device == "auto" else device
    key = f"{spec.weights}|{dev}"
    predictor = _ipa8_single_cache.get(key)
    if predictor is None:
        cfg_path = spec.weights.parent / "ipa8_config.json" if spec.weights else None
        cfg = IPA6Config.load(cfg_path)
        cfg.ensemble = None
        cfg.min_votes = 2
        cfg.device = dev
        predictor = IPA6Predictor(str(spec.weights), cfg)
        _ipa8_single_cache[key] = predictor
    bgr = _load_image(image_path)
    h, w = bgr.shape[:2]
    det = predictor.predict(bgr)
    overlay = bgr.copy()
    detections: list[dict] = []
    for i, (box, score) in enumerate(zip(det["boxes"], det["conf"])):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        detections.append(
            {
                "id": f"det_{i}",
                "x": float(((x1 + x2) / 2) / max(w, 1)),
                "y": float(((y1 + y2) / 2) / max(h, 1)),
                "class_id": 0,
                "confidence": float(score),
                "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
            }
        )
    n_inf = int(det["count"])
    cv2.putText(overlay, f"{spec.name} infected={n_inf}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return overlay, n_inf, 0, detections


def run_ipa8(spec: ModelSpec, image_path: Path, device: str) -> tuple[np.ndarray, int, int, list[dict], dict]:
    """IPA8.0 release: checksum-verified 2-model tiled ensemble (ipa4/ipa8_infer.py), loaded once per process."""
    from ipa8_infer import IPA8Model

    dev = "" if not device or device == "auto" else device
    model = IPA8Model.get(spec.weights.parent, dev)
    bgr = _load_image(image_path)
    res = model.predict(bgr)
    return model.draw(bgr, res), res["count"], 0, res["detections"], res["meta"]


def run_unet(spec: ModelSpec, image_path: Path, work_dir: Path) -> tuple[np.ndarray, int, int, list[dict]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(ML_DIR / "visualize_inference.py"),
        str(image_path),
        "--checkpoint",
        str(spec.checkpoint),
        "--out-dir",
        str(work_dir),
        "--auto-background",
    ]
    subprocess.run(cmd, cwd=str(REPO), check=True, capture_output=True)
    overlay_path = work_dir / f"{image_path.stem}_segmented.png"
    overlay = cv2.imread(str(overlay_path))
    if overlay is None:
        raise FileNotFoundError(overlay_path)
    count = 0
    summary_path = work_dir / "batch_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            count = int(rows[0].get("infected_cell_count_estimate", 0))
    return overlay, count, 0, []


def analyze_image(
    spec: ModelSpec,
    image_path: Path,
    *,
    work_dir: Path,
    conf: float | None = None,
    device: str | None = None,
    restitch_mosaic: bool | None = None,
) -> dict:
    t0 = time.perf_counter()
    conf = conf if conf is not None else settings.inference_conf
    device = device or settings.inference_device
    do_restitch = settings.mosaic_restitch_enabled if restitch_mosaic is None else bool(restitch_mosaic)
    detect_path, restitch_meta = _apply_mosaic_restitch(image_path, work_dir, do_restitch)

    raw = _load_image(detect_path)
    left_title = "Restitched" if restitch_meta.get("used_restitched") else "Original"
    model_meta = {"model": spec.name, "model_id": spec.id}

    if spec.kind == "yolo":
        overlay, n_inf, n_art, detections = run_yolo(spec, detect_path, conf, device)
        title = f"{spec.name}  |  infected={n_inf}"
    elif spec.kind == "fusion":
        tmp = work_dir / "_fusion_overlay.jpg"
        overlay, n_inf, n_art, detections = run_fusion(spec, detect_path, conf, tmp)
        title = f"{spec.name}  |  infected={n_inf} (large only)"
    elif spec.kind == "unet":
        overlay, n_inf, n_art, detections = run_unet(spec, detect_path, work_dir / "unet_work")
        title = f"{spec.name}  |  count={n_inf}"
    elif spec.kind == "ipa8":
        overlay, n_inf, n_art, detections, model_meta = run_ipa8(spec, detect_path, device)
        title = f"{spec.name}  |  infected={n_inf}"
    elif spec.kind == "ipa8_single":
        overlay, n_inf, n_art, detections = run_ipa8_single(spec, detect_path, device)
        title = f"{spec.name}  |  infected={n_inf}"
    elif spec.kind == "ipa6":
        overlay, n_inf, n_art, detections = run_ipa6(spec, detect_path, device)
        title = f"{spec.name}  |  infected={n_inf}"
    else:
        raise ValueError(spec.kind)

    collage = make_collage(raw, overlay, left_title=left_title, right_title=title)
    ms = (time.perf_counter() - t0) * 1000.0

    out_overlay = work_dir / "overlay.jpg"
    out_collage = work_dir / "collage.jpg"
    cv2.imwrite(str(out_overlay), overlay)
    cv2.imwrite(str(out_collage), collage)

    return {
        "infected_count": n_inf,
        "artifact_count": n_art,
        "processing_ms": round(ms, 1),
        "overlay_path": out_overlay,
        "collage_path": out_collage,
        "ai_detections": detections,
        "model_meta": model_meta,
        "restitch": restitch_meta,
        "warnings": list(restitch_meta.get("warnings") or []),
    }


def run_model(
    model_id: str,
    image_path: Path,
    work_dir: Path,
    *,
    restitch_mosaic: bool | None = None,
) -> dict:
    spec = get_model(model_id)
    return analyze_image(spec, image_path, work_dir=work_dir, restitch_mosaic=restitch_mosaic)
