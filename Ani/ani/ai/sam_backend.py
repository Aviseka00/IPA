"""micro-sam adapter: Segment-Anything fine-tuned for microscopy (optional).

Install with install_ai.bat, or:  pip install micro_sam  plus MobileSAM (needed for the small vit_t models):
    pip install https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip
The small 'vit_t_lm' model is used because it is practical on CPU; its weights download once on first use.
Only click-to-segment is offered; use Classic or Cellpose for whole images.
API checked against micro_sam 1.8: segment_from_points takes (row, col) points and flips them for SAM itself.
"""
from __future__ import annotations

import numpy as np

from .base import Backend, to_uint8


class MicroSamBackend(Backend):
    key = "micro_sam"
    name = "micro-sam (AI click)"
    module = "micro_sam"
    install_hint = "run install_ai.bat (needs Python 3.11+)"
    can_auto = False

    def __init__(self, model_type: str = "vit_t_lm"):
        self.model_type = model_type
        self._predictor = None
        self._cache_key = None
        self._embeddings = None

    def available(self):
        import importlib.util

        ok, msg = super().available()
        if ok and self.model_type.startswith("vit_t") and importlib.util.find_spec("mobile_sam") is None:
            return False, "MobileSAM missing – run install_ai.bat again"
        return ok, msg

    def _prepare(self, image: np.ndarray):
        self.require()
        from micro_sam import util

        if self._predictor is None:
            self._predictor = util.get_sam_model(model_type=self.model_type, device="cpu")
        key = (image.shape, float(image.ravel()[:: max(1, image.size // 997)].sum()))
        if key != self._cache_key:  # the slow step: once per image, then clicks are fast
            self._embeddings = util.precompute_image_embeddings(self._predictor, to_uint8(image), ndim=2)
            self._cache_key = key

    def segment_at(self, image, point, labels, params=None):
        from micro_sam.prompt_based_segmentation import segment_from_points

        self._prepare(image)
        if labels[tuple(int(c) for c in point)] != 0:
            return None, None
        mask = segment_from_points(self._predictor, points=np.array([[point[0], point[1]]], float),
                                   labels=np.array([1]), image_embeddings=self._embeddings)
        mask = np.squeeze(np.asarray(mask)).astype(bool) & (labels == 0)
        if not mask.any():
            return None, None
        box = tuple(slice(0, n) for n in labels.shape)
        return mask, box
