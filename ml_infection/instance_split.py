"""Split predicted infection regions into instances (watershed + optional opening)."""
from __future__ import annotations

import math

import numpy as np


def refine_binary_mask(bin_mask: np.ndarray, opening_ksize: int = 3) -> np.ndarray:
    """Morphological opening breaks thin bridges between merged regions."""
    import cv2

    m = (bin_mask > 0).astype(np.uint8)
    if opening_ksize < 2:
        return m
    k = max(3, opening_ksize | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)


def filter_instance_labels(labels: np.ndarray, min_area_px: int) -> tuple[np.ndarray, int]:
    """Remove small instances; renumber 1..N."""
    import cv2

    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for lid in sorted(np.unique(labels)):
        if lid == 0:
            continue
        area = int(np.sum(labels == lid))
        if area >= min_area_px:
            out[labels == lid] = next_id
            next_id += 1
    return out, next_id - 1


def labels_connected_components_filtered(bin_mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, int]:
    import cv2

    bin8 = (bin_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == i] = next_id
            next_id += 1
    return out, next_id - 1


def labels_true_positives_filtered(
    bin_mask: np.ndarray,
    *,
    min_area_px: int,
    border_margin_px: int = 6,
    border_touch_keep_min_area_px: int = 500,
) -> tuple[np.ndarray, int, list[tuple[int, int, int]]]:
    """
    From the thresholded binary mask, classify instances into counted positives vs artifacts.

    - Drops components below ``min_area_px`` (noise).
    - If ``border_margin_px`` > 0, detects components whose bbox touches that outer strip.
      **Small** border-touching blobs (area < ``border_touch_keep_min_area_px``) are treated
      as frame/shading artifacts. **Large** border-touching blobs (e.g. real cells cut by
      the field edge) are **kept** and counted.

    Returns ``labels`` (only true positives, renumbered 1..N), ``count``, and
    ``artifact_circles`` as ``(cx, cy, radius)`` in pixel coords for drawing red rings.
    """
    import cv2

    bin8 = (bin_mask > 0).astype(np.uint8)
    h, w = bin8.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    out = np.zeros((h, w), dtype=np.int32)
    artifact_circles: list[tuple[int, int, int]] = []
    next_id = 1
    margin = max(0, int(border_margin_px))
    edge_keep = max(int(min_area_px), int(border_touch_keep_min_area_px))

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        right = left + bw
        bottom = top + bh

        touches_border = (
            left < margin or top < margin or right > w - margin or bottom > h - margin
        )
        too_small = area < int(min_area_px)
        # Small fragments on the frame edge only (not large partial true cells).
        border_speck = touches_border and margin > 0 and area < edge_keep

        if too_small or border_speck:
            mask_i = (labels == i).astype(np.uint8)
            mu = cv2.moments(mask_i)
            if mu["m00"] > 1e-6:
                cx = int(mu["m10"] / mu["m00"])
                cy = int(mu["m01"] / mu["m00"])
            else:
                cx = left + bw // 2
                cy = top + bh // 2
            r = max(4, min(int(round(math.sqrt(area / math.pi))), max(h, w) // 8))
            artifact_circles.append((cx, cy, r))
            continue

        out[labels == i] = next_id
        next_id += 1

    return out, next_id - 1, artifact_circles


def labels_watershed_split(
    bin_mask: np.ndarray,
    *,
    min_area_px: int = 40,
    min_peak_distance_px: int = 14,
    opening_ksize: int = 5,
) -> tuple[np.ndarray, int]:
    """
    Distance-transform watershed with peak markers.
    Falls back to filtered CC if scipy/skimage unavailable or no peaks.
    """
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError:
        return labels_connected_components_filtered(bin_mask, min_area_px)

    m = refine_binary_mask(bin_mask, opening_ksize).astype(bool)
    if not m.any():
        return np.zeros(bin_mask.shape, dtype=np.int32), 0

    # Near-full-image positives (common misfire on plate scans) create huge peak counts and
    # multi-minute watershed runs; connected components is fast and still reflects "one blob".
    fg_frac = float(m.mean())
    if fg_frac > 0.38:
        return labels_connected_components_filtered(bin_mask, min_area_px)

    distance = ndi.distance_transform_edt(m)
    md = max(3, int(min_peak_distance_px))
    coords = peak_local_max(
        distance,
        min_distance=md,
        labels=m.astype(np.uint8),
        exclude_border=False,
        num_peaks=2048,
    )
    if coords.size == 0:
        return labels_connected_components_filtered(bin_mask, min_area_px)

    markers = np.zeros(m.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[int(r), int(c)] = i

    labels = watershed(-distance.astype(np.float64), markers, mask=m)
    labels_filt, count = filter_instance_labels(labels.astype(np.int32), min_area_px)
    return labels_filt, count


def choose_labels_for_counting(
    bin_mask: np.ndarray,
    *,
    min_area_px: int,
    use_watershed: bool,
    watershed_peak_distance: int,
    opening_ksize: int,
) -> tuple[np.ndarray, int]:
    if use_watershed:
        labels, n = labels_watershed_split(
            bin_mask,
            min_area_px=min_area_px,
            min_peak_distance_px=watershed_peak_distance,
            opening_ksize=opening_ksize,
        )
        if n > 0:
            return labels, n
    labels, n = labels_connected_components_filtered(bin_mask, min_area_px)
    return labels, n
