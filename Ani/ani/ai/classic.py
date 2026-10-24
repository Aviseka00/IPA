"""Built-in segmentation with no AI libraries: smoothing + threshold + distance watershed.

Runs anywhere (even very old laptops), works offline, and can be *tuned* on your corrected images
(see ani.train.tune_classic) so it improves as you annotate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from .base import Backend, click_window, pick_object, to_float, typical_diameter

DEFAULT_PARAMS = {
    "sigma": 1.0,            # smoothing (pixels)
    "threshold": "otsu",     # otsu | li | triangle | mean
    "offset": 1.0,           # multiply threshold (lower = bigger objects)
    "polarity": "auto",      # auto | bright | dark
    "split": 0.7,            # 0 = never split touching cells, 1 = split aggressively
    "min_size": 30,          # pixels (voxels for 3-D)
}

_THRESHOLDS = ("otsu", "li", "triangle", "mean")


def _threshold(values: np.ndarray, method: str) -> float:
    from skimage import filters

    fn = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
          "triangle": filters.threshold_triangle, "mean": filters.threshold_mean}[method]
    return float(fn(values))


def _sample(img: np.ndarray) -> np.ndarray:
    step = max(1, int(math.sqrt(img.size / 1_000_000)))  # threshold on ≤ ~1 M pixels for speed
    return img[(slice(None, None, step),) * img.ndim].ravel()


def foreground(image: np.ndarray, params: dict) -> np.ndarray:
    from skimage.filters import gaussian

    p = {**DEFAULT_PARAMS, **(params or {})}
    img = gaussian(to_float(image), sigma=float(p["sigma"]), preserve_range=True)
    sample = _sample(img)
    try:
        thr = _threshold(sample, p["threshold"])
    except ValueError:  # flat image
        return np.zeros(img.shape, bool)
    polarity = p["polarity"]
    if polarity == "auto":  # objects are usually the minority of pixels
        polarity = "bright" if (sample > thr).mean() <= 0.5 else "dark"
    if polarity == "dark":           # dark objects on a bright background (brightfield)
        img = img.max() - img
        sample = _sample(img)
        thr = _threshold(sample, p["threshold"])
    lo = float(np.percentile(sample, 1))
    thr = lo + (thr - lo) * float(p["offset"])
    fg = img > thr
    fg = ndi.binary_opening(fg, iterations=1)
    return ndi.binary_fill_holes(fg) if fg.ndim == 2 else fg


def split_touching(fg: np.ndarray, split: float, min_size: int) -> np.ndarray:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    comps, n = ndi.label(fg)
    if n == 0:
        return comps.astype(np.int32)
    dist = ndi.distance_transform_edt(fg)
    if split <= 0:
        return comps.astype(np.int32)
    # typical radius = median of each component's maximum distance
    radii = ndi.maximum(dist, comps, index=np.arange(1, n + 1))
    r = float(np.median(radii)) if len(radii) else 3.0
    min_distance = max(2, int(round(r * (1.6 - float(split)))))
    smooth = ndi.gaussian_filter(dist, sigma=max(0.5, r * 0.15))
    peaks = peak_local_max(smooth, min_distance=min_distance, labels=comps, exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # components without a peak (tiny) still get one marker
    missing = np.setdiff1d(np.arange(1, n + 1), np.unique(comps[markers > 0]))
    if len(missing):
        com = ndi.maximum_position(dist, comps, index=missing)
        start = len(peaks) + 1
        for k, pos in enumerate(com):
            markers[pos] = start + k
    return watershed(-smooth, markers, mask=fg)


class ClassicBackend(Backend):
    key = "classic"
    name = "Classic (built-in, no AI)"

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def auto_segment(self, image, params=None):
        from skimage.segmentation import relabel_sequential

        from .. import tools

        p = {**self.params, **(params or {})}
        fg = foreground(image, p)
        labels = split_touching(fg, float(p["split"]), int(p["min_size"]))
        labels = tools.remove_small(labels, int(p["min_size"]))
        out = relabel_sequential(labels)[0]
        return out.astype(np.uint16 if out.max() < 65535 else np.uint32)

    def segment_at(self, image, point, labels, params=None):
        """'Magic wand': threshold a window around the click, split from touching neighbours."""
        from skimage.filters import gaussian

        p = {**self.params, **(params or {})}
        diam = float(p.get("diameter") or typical_diameter(labels))
        box = click_window(labels.shape, point, diam)
        local = tuple(int(c) - b.start for c, b in zip(point, box))
        crop = gaussian(to_float(image[box]), sigma=float(p["sigma"]), preserve_range=True)
        free = labels[box] == 0
        if not free[local]:
            return None, None
        ref = float(np.percentile(crop[free], 10))          # background level
        yy = np.indices(crop.shape)
        near = np.sqrt(sum((g - c) ** 2 for g, c in zip(yy, local))) <= max(2.0, diam / 4)
        centre = float(np.median(crop[near & free]))        # brightness of the clicked object
        if centre < float(np.median(crop[free])):           # dark object (e.g. brightfield): invert
            crop = crop.max() - crop
            ref = float(np.percentile(crop[free], 10))
            centre = float(np.median(crop[near & free]))
        if centre <= ref:
            return None, None
        # half-maximum boundary: follows the real edge of blurred objects and stops at dim
        # borders between touching cells (Otsu over a mostly-background window is too low)
        thr = ref + float(p.get("wand_level", 0.5)) * (centre - ref)
        fg = (crop > thr) & free
        fg = ndi.binary_opening(fg)
        fg = ndi.binary_fill_holes(fg) & free
        # same touching-cell splitting as auto_segment, so tuned parameters help here too
        seg = split_touching(fg, float(p["split"]), int(p["min_size"]))
        mask = pick_object(seg, local, free, max_dist=diam / 2)
        if mask is None:
            return None, None
        if mask.sum() < max(5, int(p["min_size"]) // 3):
            return None, None
        return mask, box
