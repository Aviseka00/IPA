"""Fake ultralytics.YOLO returning segmentation masks like the real Results object."""
import numpy as np

__version__ = "8.0.0-fake"
CALLS = []


class _T:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Masks:
    def __init__(self, data):
        self.data = _T(data)


class _Boxes:
    def __init__(self, conf):
        self.conf = _T(conf)


class _Result:
    def __init__(self, masks, conf):
        self.masks = _Masks(masks) if len(masks) else None
        self.boxes = _Boxes(conf)


class YOLO:
    def __init__(self, model="yolo11n-seg.pt", task=None, verbose=False):
        CALLS.append(("YOLO", {"model": model}))

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        from ani.ai.classic import ClassicBackend
        img = np.asarray(source)
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3
        CALLS.append(("predict", {k: v for k, v in kwargs.items()}))
        lab = ClassicBackend().auto_segment(img.mean(-1))
        ids = [i for i in np.unique(lab) if i]
        masks = np.stack([(lab == i).astype(np.float32) for i in ids]) if ids else np.zeros((0,) + lab.shape)
        return [_Result(masks, np.linspace(0.9, 0.5, len(ids)))]
