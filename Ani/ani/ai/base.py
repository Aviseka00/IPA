"""Common interface for every segmentation backend (built-in or optional AI package)."""
from __future__ import annotations

import importlib.util

import numpy as np


class BackendUnavailable(RuntimeError):
    pass


class Backend:
    key = "base"
    name = "Base"
    install_hint = ""            # shown when the package is missing
    can_auto = True              # can label a whole image
    can_click = True             # can segment one object from a click
    module = None                # python module that must be importable

    def available(self) -> tuple[bool, str]:
        if self.module is None or importlib.util.find_spec(self.module) is not None:
            return True, "ready"
        return False, f"not installed – {self.install_hint}"

    def require(self):
        ok, msg = self.available()
        if not ok:
            raise BackendUnavailable(f"{self.name}: {msg}")

    def auto_segment(self, image: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Label every object in a single-channel 2-D (or 3-D) image."""
        raise NotImplementedError(f"{self.name} cannot label whole images")

    def segment_at(self, image: np.ndarray, point, labels: np.ndarray, params: dict | None = None):
        """Segment the one object under `point` in a 2-D image.

        Returns (mask, box): boolean mask for the crop `labels[box]`, never covering existing labels,
        or (None, None) when nothing was found."""
        raise NotImplementedError(f"{self.name} cannot segment from a click")


# ------------------------------------------------------------------ helpers shared by backends
def to_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    img = to_float(image)
    lo, hi = np.percentile(img[:: max(1, img.shape[0] // 512)], (0.5, 99.8))
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def typical_diameter(labels: np.ndarray, default: float = 20.0) -> float:
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    if not len(areas):
        return default
    if labels.ndim == 3:
        return float(2 * (3 * np.median(areas) / (4 * np.pi)) ** (1 / 3))
    return float(2 * np.sqrt(np.median(areas) / np.pi))


def polarity_from_labels(image: np.ndarray, labels: np.ndarray) -> str | None:
    """'bright' if labelled objects are brighter than the rest of the image, 'dark' if darker, None if unknown."""
    if image is None or labels is None or image.shape != labels.shape:
        return None
    fg = labels > 0
    if fg.sum() < 20 or (~fg).sum() < 20:
        return None
    img = np.asarray(image, dtype=np.float32)
    return "bright" if float(img[fg].mean()) >= float(img[~fg].mean()) else "dark"


def click_window(shape, point, diameter: float, factor: float = 3.0, minimum: int = 48):
    """Square crop around a click, about `factor` cell diameters wide."""
    half = int(max(minimum, factor * diameter))
    return tuple(slice(max(int(p) - half, 0), min(int(p) + half + 1, n)) for p, n in zip(point, shape))


def pick_object(seg: np.ndarray, local_point, free: np.ndarray, max_dist: float):
    """Object of `seg` under the click (or the nearest one within max_dist), limited to free pixels."""
    from scipy import ndimage as ndi

    lab = int(seg[tuple(local_point)])
    if lab == 0:
        if not seg.any():
            return None
        dist, idx = ndi.distance_transform_edt(seg == 0, return_indices=True)
        if dist[tuple(local_point)] > max_dist:
            return None
        lab = int(seg[tuple(i[tuple(local_point)] for i in idx)])
    mask = (seg == lab) & free
    if not mask.any():
        return None
    # keep only the connected piece nearest the click
    comp, n = ndi.label(mask)
    if n > 1:
        c = comp[tuple(local_point)]
        if c == 0:
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            c = int(np.argmax(sizes))
        mask = comp == c
    return ndi.binary_fill_holes(mask) & free
