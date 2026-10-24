"""Auto-detect gel / blot lanes and bands for densitometry.

Classical detectors inspired by ImageJ lane finding and GelGenie-style band
boxes — no QuPath/PyTorch runtime required.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .segment import to_gray


def find_lanes(image: np.ndarray, max_lanes: int = 12) -> list[tuple[int, int, int, int]]:
    """Return lane rectangles as ``(row0, col0, row1, col1)`` for vertical blot lanes.

    Works on dark bands on a brighter gel background. Returns an empty list when
    the picture does not look like a multi-lane gel.
    """
    gray = np.asarray(to_gray(image), np.float32)
    if gray.ndim != 2 or min(gray.shape) < 32:
        return []
    height, width = gray.shape
    profile = gray.mean(axis=0)
    smooth = ndi.gaussian_filter1d(profile, sigma=max(1.0, width / 200.0))
    inverted = float(smooth.max()) - smooth
    if float(inverted.max() - inverted.min()) < 8:
        return []
    distance = max(8, width // max(max_lanes * 2, 4))
    prominence = max(3.0, 0.12 * float(inverted.max() - inverted.min()))
    peaks, props = find_peaks(inverted, distance=distance, prominence=prominence)
    if peaks.size < 2:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_lanes]
    centers = np.sort(peaks[order])
    bounds = [0]
    for left, right in zip(centers[:-1], centers[1:]):
        bounds.append(int((int(left) + int(right)) // 2))
    bounds.append(width)
    lanes: list[tuple[int, int, int, int]] = []
    margin = max(2, height // 40)
    for i in range(len(bounds) - 1):
        c0, c1 = int(bounds[i]), int(bounds[i + 1])
        if c1 - c0 < 6:
            continue
        strip = gray[:, c0:c1]
        row_mean = strip.mean(axis=1)
        active = row_mean < (float(row_mean.max()) - 0.15 * float(row_mean.max() - row_mean.min()))
        if not np.any(active):
            r0, r1 = margin, height - margin
        else:
            rows = np.flatnonzero(active)
            r0 = max(0, int(rows[0]) - margin)
            r1 = min(height, int(rows[-1]) + margin + 1)
        if r1 - r0 < 8:
            continue
        lanes.append((r0, c0, r1, c1))
    return lanes


def find_band_rows(image: np.ndarray, max_rows: int = 8) -> list[tuple[int, int, int, int]]:
    """Horizontal blot rows (western-style): one wide ROI per band row."""
    gray = np.asarray(to_gray(image), np.float32)
    if gray.ndim != 2 or min(gray.shape) < 24:
        return []
    height, width = gray.shape
    profile = gray.mean(axis=1)
    smooth = ndi.gaussian_filter1d(profile, sigma=max(1.0, height / 80.0))
    inverted = float(smooth.max()) - smooth
    if float(inverted.max() - inverted.min()) < 5:
        return []
    distance = max(6, height // max(max_rows * 2, 4))
    prominence = max(2.0, 0.1 * float(inverted.max() - inverted.min()))
    peaks, props = find_peaks(inverted, distance=distance, prominence=prominence)
    if peaks.size < 1:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_rows]
    centers = np.sort(peaks[order])
    rows: list[tuple[int, int, int, int]] = []
    margin_c = max(2, width // 40)
    for center in centers:
        # Tight vertical window around the band row (not half the gel).
        half = max(8, int(height / (2 * max(len(centers) + 1, 3))))
        r0 = max(0, int(center) - half)
        r1 = min(height, int(center) + half + 1)
        strip = gray[r0:r1, :]
        col_mean = strip.mean(axis=0)
        active = col_mean < (float(col_mean.max()) - 0.1 * float(col_mean.max() - col_mean.min()))
        if not np.any(active):
            c0, c1 = margin_c, width - margin_c
        else:
            cols = np.flatnonzero(active)
            c0 = max(0, int(cols[0]) - margin_c)
            c1 = min(width, int(cols[-1]) + margin_c + 1)
        if c1 - c0 < 12 or r1 - r0 < 6:
            continue
        rows.append((r0, c0, r1, c1))
    rows.sort(key=lambda box: box[0])
    return rows


def find_bands(image: np.ndarray, max_bands: int = 48) -> list[tuple[int, int, int, int]]:
    """Individual dark blot/gel bands as tight ROIs (classical GelGenie-style boxes)."""
    gray = np.asarray(to_gray(image), np.float32)
    if gray.ndim != 2 or min(gray.shape) < 24:
        return []
    height, width = gray.shape
    inverted = float(gray.max()) - gray
    smooth = ndi.gaussian_filter(inverted, sigma=1.15)
    floor = float(np.percentile(smooth, 65))
    span = float(smooth.max() - floor)
    if span < 2.5:
        return []
    # Low threshold catches faint left-side blot bands; opening drops speckles.
    mask = smooth >= (floor + 0.08 * span)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)
    labeled, count = ndi.label(mask)
    if count < 1:
        return []
    slices = ndi.find_objects(labeled)
    bands: list[tuple[int, int, int, int]] = []
    min_area = max(18, int(0.0004 * height * width))
    max_area = int(0.12 * height * width)
    for index, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        r0, r1 = int(slc[0].start), int(slc[0].stop)
        c0, c1 = int(slc[1].start), int(slc[1].stop)
        area = (r1 - r0) * (c1 - c0)
        if area < min_area or area > max_area:
            continue
        bh, bw = r1 - r0, c1 - c0
        # Reject huge background blobs and hairline noise.
        if max(bh, bw) < 4:
            continue
        if bh > 0.45 * height and bw > 0.45 * width:
            continue
        component = labeled[r0:r1, c0:c1] == index
        if int(component.sum()) < min_area // 2:
            continue
        # Pad so densitometry has a rim for background.
        pad_r = max(1, bh // 6)
        pad_c = max(1, bw // 6)
        bands.append((
            max(0, r0 - pad_r),
            max(0, c0 - pad_c),
            min(height, r1 + pad_r),
            min(width, c1 + pad_c),
        ))
    bands.sort(key=lambda box: ((box[0] + box[2]) // 2 // max(8, height // 20), (box[1] + box[3]) // 2))
    return bands[:max_bands]


def find_gel_rois(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Best automatic ROIs: blot rows when landscape, else vertical lanes, else bands."""
    gray = np.asarray(to_gray(image), np.float32)
    height, width = gray.shape[:2] if gray.ndim == 2 else (0, 0)
    rows = find_band_rows(image)
    lanes = find_lanes(image)
    # Western-style: wide image + clear horizontal band rows win over vertical splits.
    if len(rows) >= 2 and (width >= height or len(lanes) < 2):
        return rows
    if len(lanes) >= 2:
        return lanes
    if len(rows) >= 1:
        return rows
    return find_bands(image)


def lanes_as_rois(lanes: list[tuple[int, int, int, int]], kind: str = "lane") -> list[dict]:
    """ROI dicts Ani's densitometry UI already understands."""
    rois = []
    for index, roi in enumerate(lanes, start=1):
        label = "band" if kind == "band" else ("row" if kind == "row" else "lane")
        rois.append({
            "kind": kind if kind in ("lane", "band", "row") else "lane",
            "name": f"{label} {index}",
            "roi": tuple(int(v) for v in roi),
            "mask": None,
            "label": None,
        })
    return rois
