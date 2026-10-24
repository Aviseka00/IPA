"""Segmentation backends. The built-in Classic backend always works; the others need optional packages.

Your own trained models are found automatically in the project folder:
    models/cellpose/<any file>      Cellpose models (from Ani's training or the Colab notebook)
    models/yolo/<name>.pt           YOLO-seg models (from the Colab notebook)
"""
from __future__ import annotations

from pathlib import Path

from .base import Backend, BackendUnavailable  # noqa: F401
from .cellpose_backend import CellposeBackend
from .classic import DEFAULT_PARAMS, ClassicBackend  # noqa: F401
from .sam_backend import MicroSamBackend
from .yolo_backend import YoloBackend


def backends_for(project=None) -> list[Backend]:
    params = (project.meta.get("classic_params") if project else None) or {}
    out: list[Backend] = [ClassicBackend(params),
                          CellposeBackend("cyto3"), CellposeBackend("nuclei"),
                          MicroSamBackend()]
    if project is not None:
        root = Path(project.root) / "models"
        for f in sorted((root / "cellpose").glob("*")) if (root / "cellpose").exists() else []:
            if f.is_file() and not f.name.startswith(".") and f.suffix not in (".json", ".txt", ".npy"):
                out.insert(1, CellposeBackend(pretrained=f))
        for f in sorted((root / "yolo").glob("*.pt")) if (root / "yolo").exists() else []:
            out.insert(1, YoloBackend(f))
    return out
