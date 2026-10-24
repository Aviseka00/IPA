"""YOLO-seg adapter for models you trained from Ani's YOLO export (optional: pip install ultralytics)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Backend, to_uint8


class YoloBackend(Backend):
    module = "ultralytics"
    install_hint = "pip install ultralytics"
    can_click = False

    def __init__(self, weights: Path):
        self.weights = Path(weights)
        self.key = f"yolo:{self.weights.name}"
        self.name = f"YOLO-seg – your model ({self.weights.name})"
        self._model = None

    def available(self):
        ok, msg = super().available()
        if ok and not self.weights.exists():
            return False, f"model file missing: {self.weights}"
        return ok, msg

    def auto_segment(self, image, params=None):
        params = params or {}
        if np.asarray(image).ndim != 2:
            raise ValueError("YOLO models work on 2-D images only")
        self.require()
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        rgb = np.repeat(to_uint8(image)[..., None], 3, axis=-1)
        res = self._model.predict(rgb, conf=float(params.get("conf", 0.25)), device="cpu",
                                  retina_masks=True, verbose=False)[0]
        h, w = rgb.shape[:2]
        out = np.zeros((h, w), np.uint16)
        if res.masks is None:
            return out
        masks = np.asarray(res.masks.data.cpu().numpy() if hasattr(res.masks.data, "cpu") else res.masks.data) > 0.5
        conf = np.asarray(res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else res.boxes.conf)
        for i, k in enumerate(np.argsort(conf), 1):    # most confident painted last (wins overlaps)
            m = masks[k]
            if m.shape != (h, w):
                from skimage.transform import resize
                m = resize(m.astype(float), (h, w), order=0) > 0.5
            out[m] = i
        from skimage.segmentation import relabel_sequential

        return relabel_sequential(out)[0].astype(np.uint16)
