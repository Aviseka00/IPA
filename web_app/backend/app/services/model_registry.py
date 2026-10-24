"""Available inference models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import settings

REPO = settings.repo_root
ML = REPO / "ml_infection"
IPA4_CFG = settings.ipa4_config
WEIGHTS = REPO / "weights"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    kind: str  # unet | yolo | fusion | ipa6 | ipa8 | ipa8_single
    weights: Path | None = None
    imgsz: int = 640
    checkpoint: Path | None = None


def list_models() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            id="ipa8",
            name="IPA8.0 (recommended)",
            description="Validated release 13 Oct 2026 — 2-model ensemble on 1024 px tiles with "
                        "rotation voting. Test F1 0.918, count error 6.9% (QC-corrected labels).",
            kind="ipa8",
            weights=WEIGHTS / "IPA8.0" / "IPA8.0_A.pt",
            imgsz=1024,
        ),
        ModelSpec(
            id="ipa8_a",
            name="IPA8.0 A (YOLOv8m)",
            description="IPA8 member A only — YOLOv8m-seg from IPA5, 1024 px tiles and 4-view voting. For comparison with the ensemble.",
            kind="ipa8_single",
            weights=WEIGHTS / "IPA8.0" / "IPA8.0_A.pt",
            imgsz=1024,
        ),
        ModelSpec(
            id="ipa8_b",
            name="IPA8.0 B (YOLOv8s)",
            description="IPA8 member B only — YOLOv8s-seg from IPA4 Phase2, 1024 px tiles and 4-view voting. For comparison with the ensemble.",
            kind="ipa8_single",
            weights=WEIGHTS / "IPA8.0" / "IPA8.0_B.pt",
            imgsz=1024,
        ),
        ModelSpec(
            id="ipav2",
            name="IPAV2 (IPA2)",
            description="Legacy U-Net segmentation — IPAV2.0",
            kind="unet",
            checkpoint=ML / "artifacts" / "IPAV2.0" / "IPAV2.0_seg_best.pt",
        ),
        ModelSpec(
            id="ipa3",
            name="IPA3.0",
            description="U-Net segmentation — IPA3.0",
            kind="unet",
            checkpoint=ML / "artifacts" / "IPA3.0" / "IPA3.0_seg_best.pt",
        ),
        ModelSpec(
            id="ipa4_phase1",
            name="IPA4 Phase1 (640)",
            description="YOLOv8 segmentation @ 640px",
            kind="yolo",
            weights=WEIGHTS / "ipa4" / "phase1_640.pt",
            imgsz=640,
        ),
        ModelSpec(
            id="ipa4_phase2",
            name="IPA4 Phase2 (1280)",
            description="YOLOv8 production model @ 1280px",
            kind="yolo",
            weights=WEIGHTS / "ipa4" / "best.pt",
            imgsz=1280,
        ),
        ModelSpec(
            id="ipa5",
            name="IPA5 (FINAL)",
            description="YOLOv8 segmentation @ 640px — latest trained model",
            kind="yolo",
            weights=WEIGHTS / "ipa4" / "ipa5_FINAL.pt",
            imgsz=640,
        ),
        ModelSpec(
            id="ipa6",
            name="IPA6 (tiled)",
            description="Development build of the tiled model — superseded by IPA8.0",
            kind="ipa6",
            weights=REPO / "ipa6" / "IPA6.pt",
            imgsz=1024,
        ),
        ModelSpec(
            id="ipa4_fusion",
            name="IPA4 Fusion",
            description="YOLO + classical rei/reb — large cells (yellow circle count)",
            kind="fusion",
            weights=WEIGHTS / "ipa4" / "best.pt",
            imgsz=1280,
        ),
    ]
    return specs


def get_model(model_id: str) -> ModelSpec:
    for m in list_models():
        if m.id == model_id:
            if m.kind == "unet" and m.checkpoint and m.checkpoint.is_file():
                return m
            if m.kind in ("yolo", "fusion", "ipa6", "ipa8", "ipa8_single") and model_available(m):
                return m
            if m.kind == "unet" and (not m.checkpoint or not m.checkpoint.is_file()):
                raise FileNotFoundError(f"Checkpoint missing for {m.name}")
            if m.kind in ("yolo", "fusion", "ipa6", "ipa8", "ipa8_single") and not model_available(m):
                raise FileNotFoundError(f"Weights missing for {m.name}")
            return m
    raise KeyError(f"Unknown model: {model_id}")


def model_available(spec: ModelSpec) -> bool:
    if spec.kind == "unet":
        return bool(spec.checkpoint and spec.checkpoint.is_file())
    if spec.kind == "ipa8":
        d = spec.weights.parent if spec.weights else None
        return bool(d and all((d / f).is_file() for f in
                              ("IPA8.0_A.pt", "IPA8.0_B.pt", "ipa8_config.json", "IPA8.0_MODEL_CARD.json")))
    if spec.kind == "ipa8_single":
        return bool(spec.weights and spec.weights.is_file())
    return bool(spec.weights and spec.weights.is_file())
