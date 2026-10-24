"""Cellpose adapter (optional: `pip install "cellpose>=3.1,<4"`, CPU is fine).

Cellpose 3 is recommended on CPU: its cyto3 / nuclei models are small (~25 MB) and quick.
Cellpose 4 (Cellpose-SAM) also works but is much slower without a GPU.
Model weights download once on first use; afterwards everything is offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Backend, click_window, pick_object, typical_diameter


def cellpose_major() -> int:
    from importlib.metadata import version

    try:
        return int(version("cellpose").split(".")[0])
    except Exception:
        return 3


class CellposeBackend(Backend):
    module = "cellpose"
    install_hint = 'run install_ai.bat (or: pip install "cellpose>=3.1,<4")'

    def __init__(self, model_type: str = "cyto3", pretrained: Path | None = None, label: str | None = None):
        self.model_type = model_type
        self.pretrained = Path(pretrained) if pretrained else None
        self.key = f"cellpose:{self.pretrained.name if self.pretrained else model_type}"
        self.name = label or (f"Cellpose – your model ({self.pretrained.name})" if self.pretrained
                              else f"Cellpose {model_type}")
        self._model = None

    def available(self):
        ok, msg = super().available()
        if ok and self.pretrained is not None and not self.pretrained.exists():
            return False, f"model file missing: {self.pretrained}"
        return ok, msg

    def _get_model(self):
        if self._model is None:
            self.require()
            from cellpose import models

            if self.pretrained is not None:
                self._model = models.CellposeModel(gpu=False, pretrained_model=str(self.pretrained))
            elif hasattr(models, "Cellpose") and cellpose_major() < 4:
                self._model = models.Cellpose(gpu=False, model_type=self.model_type)   # estimates cell size
            else:
                self._model = models.CellposeModel(gpu=False)                         # Cellpose-SAM (v4)
        return self._model

    def _eval(self, image: np.ndarray, diameter: float | None, params: dict) -> np.ndarray:
        model = self._get_model()
        kwargs = {"diameter": diameter or None,
                  "flow_threshold": float(params.get("flow_threshold", 0.4)),
                  "cellprob_threshold": float(params.get("cellprob_threshold", 0.0))}
        if cellpose_major() < 4:
            kwargs["channels"] = [0, 0]           # single grey channel
        if image.ndim == 3:                       # z-stack: 2-D per slice, stitched (much faster than 3-D on CPU)
            kwargs.update(z_axis=0, stitch_threshold=float(params.get("stitch_threshold", 0.3)))
        out = model.eval(image, **kwargs)
        masks = out[0] if isinstance(out, (tuple, list)) else out
        masks = np.asarray(masks)
        return masks.astype(np.uint16 if masks.max() < 65535 else np.uint32)

    def auto_segment(self, image, params=None):
        params = params or {}
        return self._eval(np.asarray(image), params.get("diameter") or None, params)

    def segment_at(self, image, point, labels, params=None):
        params = params or {}
        diam = float(params.get("diameter") or typical_diameter(labels, default=30.0))
        box = click_window(labels.shape, point, diam, factor=2.5, minimum=64)
        local = tuple(int(c) - b.start for c, b in zip(point, box))
        free = labels[box] == 0
        if not free[local]:
            return None, None
        seg = self._eval(np.asarray(image[box]), diam, params)
        mask = pick_object(seg, local, free, max_dist=diam / 2)
        return (mask, box) if mask is not None else (None, None)
