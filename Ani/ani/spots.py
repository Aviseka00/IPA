"""Small dark infected cells on a bright field, including a second copy of the same cell.

Some of these scans show the same cell twice, shifted by a few dozen pixels.
Each dark spot is labeled. The extra copy is marked duplicate.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

from .segment import to_gray

SPOT_CAP = 1500
_DISK_OFF = None


def find_infected(image: np.ndarray) -> tuple[np.ndarray, dict[int, str], dict]:
    """Label each dark spot. Returns ``(labels, {id: class}, meta)``.

    ``meta`` has ``kept``, ``capped``, ``pairs``, and ``seconds``.
    """
    started = time.perf_counter()
    gray = np.asarray(to_gray(image), np.float32)
    if not gray.flags["C_CONTIGUOUS"]:
        gray = np.ascontiguousarray(gray)
    labels = np.zeros(gray.shape, np.uint16)
    empty = {"kept": 0, "capped": False, "pairs": {}, "seconds": 0.0}
    if gray.ndim != 2 or min(gray.shape) < 32:
        empty["seconds"] = time.perf_counter() - started
        return labels, {}, empty

    if min(gray.shape) >= 256:
        preview = gray[::2, ::2]
        smooth = ndi.uniform_filter(preview, size=15)
        field_small = ndi.binary_erosion(smooth > 80, iterations=6)
        field = np.repeat(np.repeat(field_small, 2, axis=0), 2, axis=1)[: gray.shape[0], : gray.shape[1]]
    else:
        smooth = ndi.uniform_filter(gray, size=31)
        field = ndi.binary_erosion(smooth > 80, iterations=12)
    if int(field.sum()) < 1000:
        field = np.ones(gray.shape, bool)

    local = ndi.uniform_filter(gray, size=11)
    contrast = local - gray
    minima = (
        (gray == ndi.minimum_filter(gray, size=5))
        & field
        & (local > 165)
        & (gray < 170)
        & (contrast >= 48)
    )
    ys, xs = np.nonzero(minima)
    if ys.size == 0:
        empty["seconds"] = time.perf_counter() - started
        return labels, {}, empty

    strengths = contrast[ys, xs]
    order = np.argsort(strengths)[::-1]
    chosen_y: list[int] = []
    chosen_x: list[int] = []
    chosen_c: list[float] = []
    # Coarse occupancy grid (~6 px) avoids writing full 13×13 masks during NMS.
    step = 6
    grid_h = (gray.shape[0] + step - 1) // step
    grid_w = (gray.shape[1] + step - 1) // step
    taken = np.zeros((grid_h, grid_w), np.uint8)
    capped = False
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        gy, gx = y // step, x // step
        if taken[gy, gx]:
            continue
        chosen_y.append(y)
        chosen_x.append(x)
        chosen_c.append(float(strengths[index]))
        y0, y1 = max(0, gy - 1), min(grid_h, gy + 2)
        x0, x1 = max(0, gx - 1), min(grid_w, gx + 2)
        taken[y0:y1, x0:x1] = 1
        if len(chosen_y) >= SPOT_CAP:
            capped = True
            break

    chosen = list(zip(chosen_y, chosen_x, chosen_c))
    duplicate, partners = _duplicate_flags(gray, chosen)
    classes: dict[int, str] = {}
    pairs: dict[int, int] = {}
    rr, cc = _disk_offsets(4)
    for number, ((y, x, _contrast), is_copy) in enumerate(zip(chosen, duplicate), start=1):
        if number > 65534:
            break
        rows = rr + int(y)
        cols = cc + int(x)
        keep = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
        labels[rows[keep], cols[keep]] = number
        classes[number] = "duplicate" if is_copy else "infected"
    for copy_index, primary_index in partners.items():
        copy_id = copy_index + 1
        primary_id = primary_index + 1
        if copy_id in classes and primary_id in classes:
            pairs[copy_id] = primary_id
            pairs[primary_id] = copy_id
    meta = {
        "kept": len(classes),
        "capped": capped,
        "pairs": pairs,
        "seconds": float(time.perf_counter() - started),
    }
    return labels, classes, meta


def _disk_offsets(radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    global _DISK_OFF
    if _DISK_OFF is None or _DISK_OFF[2] != radius:
        yy, xx = np.mgrid[-radius: radius + 1, -radius: radius + 1]
        mask = yy * yy + xx * xx <= radius * radius
        _DISK_OFF = (yy[mask].astype(np.int32), xx[mask].astype(np.int32), radius)
    return _DISK_OFF[0], _DISK_OFF[1]


def _duplicate_flags(gray: np.ndarray, spots: list[tuple[int, int, float]]) -> tuple[list[bool], dict[int, int]]:
    """A spot is a duplicate when a stronger spot nearby has the same appearance."""
    flags = [False] * len(spots)
    partners: dict[int, int] = {}
    if len(spots) < 2:
        return flags, partners
    points = np.asarray([(y, x) for y, x, _c in spots], np.float32)
    try:
        from scipy.spatial import cKDTree

        neighbors = cKDTree(points).query_ball_point(points, r=42)
    except Exception:
        neighbors = [list(range(len(spots))) for _ in spots]
    fill = float(np.median(gray[::16, ::16])) if gray.size else 0.0
    patches = _extract_patches(gray, spots, fill)
    flat = patches.reshape(len(spots), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    claimed = set()
    strength = [c for _y, _x, c in spots]
    for index, near in enumerate(neighbors):
        if index in claimed:
            continue
        left_norm = float(norms[index])
        if left_norm < 1e-6:
            continue
        best = None
        best_score = 0.72
        left = flat[index]
        py, px = points[index]
        for other in near:
            if other == index or other in claimed:
                continue
            dy = float(points[other, 0] - py)
            dx = float(points[other, 1] - px)
            distance = float(np.hypot(dy, dx))
            if distance < 16:
                continue
            right_norm = float(norms[other])
            if right_norm < 1e-6:
                continue
            score = float(np.dot(left, flat[other]) / (left_norm * right_norm))
            # Prefer the common ~30 px vertical stitch twin when scores are close.
            if abs(abs(dy) - 30.0) < 6.0 and abs(dx) < 4.0:
                score += 0.03
            if score > best_score:
                best_score = score
                best = other
        if best is None:
            continue
        copy = index if strength[index] <= strength[best] else best
        primary = best if copy == index else index
        flags[copy] = True
        partners[copy] = primary
        claimed.add(index)
        claimed.add(best)
    return flags, partners


def _extract_patches(gray: np.ndarray, spots: list[tuple[int, int, float]], fill: float) -> np.ndarray:
    patches = np.full((len(spots), 13, 13), float(fill), np.float32)
    height, width = gray.shape
    for i, (y, x, _c) in enumerate(spots):
        y0, y1 = max(0, y - 6), min(height, y + 7)
        x0, x1 = max(0, x - 6), min(width, x + 7)
        patches[i, y0 - (y - 6): y0 - (y - 6) + (y1 - y0), x0 - (x - 6): x0 - (x - 6) + (x1 - x0)] = gray[y0:y1, x0:x1]
    return patches
