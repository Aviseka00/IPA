"""Mask correction operations (inspired by deepcell-label's edit tools, written fresh).

All functions take a label array and return a NEW array; they work for 2-D and 3-D.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def next_free_label(labels: np.ndarray) -> int:
    return int(labels.max()) + 1 if labels.size else 1


def label_ids(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    return ids[ids != 0]


def delete_label(labels: np.ndarray, label: int) -> np.ndarray:
    out = labels.copy()
    if label:
        out[out == label] = 0
    return out


def _bbox(mask: np.ndarray, pad: int = 3):
    sl = ndi.find_objects(mask.astype(np.uint8))[0]
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, mask.shape))


def merge_labels(labels: np.ndarray, keep: int, other: int) -> np.ndarray:
    """Merge `other` into `keep` and fill the thin gap between them (works on a crop, so it is fast)."""
    out = labels.copy()
    if not keep or not other or keep == other:
        return out
    union = (out == keep) | (out == other)
    if not union.any():
        return out
    box = _bbox(union)
    sub, u = out[box], union[box]
    closed = ndi.binary_closing(u, structure=np.ones((3,) * labels.ndim), iterations=2)
    sub[(closed & (sub == 0)) | u] = keep
    return out


def split_label(labels: np.ndarray, label: int, parts: int = 2) -> np.ndarray:
    """Split one cell into `parts` pieces with a distance-transform watershed."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    out = labels.copy()
    region = out == label
    if not label or region.sum() < 4 * parts:
        return out
    slices = ndi.find_objects(region.astype(np.uint8))[0]
    pad = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in slices)
    sub = region[pad]
    dist = ndi.distance_transform_edt(sub)
    peaks = peak_local_max(dist, min_distance=2, num_peaks=parts, labels=sub.astype(int), exclude_border=False)
    if len(peaks) < 2:  # round blob without two peaks: split along its long axis instead
        coords = np.argwhere(sub)
        centred = coords - coords.mean(0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        proj = centred @ axis
        edges = np.quantile(proj, np.linspace(0, 1, parts + 1)[1:-1])
        piece = np.digitize(proj, edges)
        sub_out = np.zeros(sub.shape, np.int64)
        sub_out[tuple(coords.T)] = piece + 1
    else:
        markers = np.zeros(sub.shape, np.int64)
        for i, p in enumerate(peaks, 1):
            markers[tuple(p)] = i
        sub_out = watershed(-dist, markers, mask=sub)
    view = out[pad]
    new_id = next_free_label(out)
    for piece_id in range(2, int(sub_out.max()) + 1):
        view[sub_out == piece_id] = new_id
        new_id += 1
    return out


def remove_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    out = labels.copy()
    ids, counts = np.unique(out, return_counts=True)
    small = ids[(counts < min_size) & (ids != 0)]
    if len(small):
        out[np.isin(out, small)] = 0
    return out


def fill_holes(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    for lid, sl in zip(range(1, int(out.max()) + 1), ndi.find_objects(out)):
        if sl is None:
            continue
        sub = out[sl]
        filled = ndi.binary_fill_holes(sub == lid)
        sub[filled & (sub == 0)] = lid
    return out


def relabel_sequential(labels: np.ndarray, classes: dict[int, str] | None = None):
    """Renumber cells 1..N (no gaps). Returns (labels, remapped classes)."""
    ids = label_ids(labels)
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=labels.dtype)
    lut[ids] = np.arange(1, len(ids) + 1)
    out = lut[labels]
    new_classes = {int(lut[o]): c for o, c in (classes or {}).items() if 0 < o < len(lut) and lut[o]}
    return out, new_classes


def compare(pred: np.ndarray | None, corrected: np.ndarray, iou_match: float = 0.5) -> dict:
    """How much did the human change the detector's output? Useful to track model quality."""
    n_corr = len(label_ids(corrected))
    if pred is None or pred.shape != corrected.shape:
        return {"cells": n_corr, "matched": 0, "added": n_corr, "removed": 0, "modified": 0}
    p_ids, c_ids = label_ids(pred), label_ids(corrected)
    both = (pred > 0) & (corrected > 0)
    base = np.int64(corrected.max()) + 1
    code = pred[both].astype(np.int64) * base + corrected[both]   # 1-D pair encoding: much faster than unique(axis=1)
    codes, inter = np.unique(code, return_counts=True)
    pairs = np.stack([codes // base, codes % base])
    p_area = dict(zip(*np.unique(pred[pred > 0], return_counts=True)))
    c_area = dict(zip(*np.unique(corrected[corrected > 0], return_counts=True)))
    order = np.argsort(-inter)  # match biggest overlaps first
    pairs, inter = pairs[:, order], inter[order]
    matched_p, matched_c, modified = set(), set(), 0
    for (pi, ci), n in zip(pairs.T, inter):
        iou = n / (p_area[pi] + c_area[ci] - n)
        if iou >= iou_match and pi not in matched_p and ci not in matched_c:
            matched_p.add(pi)
            matched_c.add(ci)
            if iou < 0.98:
                modified += 1
    return {
        "cells": n_corr,
        "matched": len(matched_c),
        "added": int(len(c_ids) - len(matched_c)),     # missed by detector
        "removed": int(len(p_ids) - len(matched_p)),   # false detections
        "modified": int(modified),                     # boundary fixed
    }


def region_info(labels: np.ndarray, label: int) -> dict:
    region = labels == label
    area = int(region.sum())
    if not area:
        return {"label": label, "area": 0}
    centroid = [round(float(c), 1) for c in np.argwhere(region).mean(0)]
    return {"label": int(label), "area": area, "centroid": centroid}


# --------------------------------------------------------------------------- review helpers
REASONS = {
    "small": "very small – debris or a fragment?",
    "large": "very large – several cells merged?",
    "irregular": "irregular shape (solidity {solidity:.2f}) – merged cells?",
    "pieces": "object is in {n} separate pieces",
    "half": "looks like half a cell – together with object {partner} it forms one round cell",
    "missed_bright": "unlabelled bright object (~{area} px) – missed cell?",
    "missed_dark": "unlabelled dark object (~{area} px) – missed cell?",
}


def _reason(issues: list[dict]) -> str:
    return "; ".join(REASONS[i["code"]].format(**i) for i in issues)


def add_new_objects(labels: np.ndarray, new: np.ndarray, max_overlap: float = 0.2) -> np.ndarray:
    """Add objects from `new` that do not already exist in `labels` (overlap < max_overlap).
    Existing objects are never changed; new ones only fill empty pixels."""
    new = np.asarray(new)
    if not new.any():
        return labels.copy()
    area = np.bincount(new.ravel())
    both = (new > 0) & (labels > 0)
    overlap = np.bincount(new[both].ravel(), minlength=len(area))
    ids = np.nonzero(area)[0]
    ids = ids[ids > 0]
    keep = ids[overlap[ids] / area[ids] < max_overlap]
    out = labels.astype(np.int64)
    if len(keep):
        lut = np.zeros(len(area), np.int64)
        start = int(labels.max()) + 1
        lut[keep] = np.arange(start, start + len(keep))
        mapped = lut[new]
        place = (mapped > 0) & (out == 0)
        out[place] = mapped[place]
    from .io import to_label_dtype
    return to_label_dtype(out)


def find_suspects(labels: np.ndarray) -> list[dict]:
    """Cells that are probably detector mistakes: fragments, giants, irregular (merged) shapes, broken pieces."""
    from skimage.measure import regionprops

    props = regionprops(labels)
    if not props:
        return []
    areas = np.array([p.area for p in props], float)
    med = float(np.median(areas))
    all_areas = np.bincount(labels.ravel())
    boxes = ndi.find_objects(labels) if labels.ndim == 2 else None
    out = []
    for p in props:
        issues, score = [], 0.0
        if len(props) >= 5 and p.area < 0.3 * med:
            issues.append({"code": "small"})
            score += 1 - p.area / (0.3 * med)
        if len(props) >= 5 and p.area > 2.5 * med:
            issues.append({"code": "large"})
            score += min(p.area / (2.5 * med), 4) - 0.5
        if labels.ndim == 2 and p.area >= 20 and p.solidity < 0.85:
            issues.append({"code": "irregular", "solidity": float(p.solidity)})
            score += (0.85 - p.solidity) * 5
        pieces = ndi.label(p.image)[1]
        if pieces > 1:
            issues.append({"code": "pieces", "n": int(pieces)})
            score += 1
        if labels.ndim == 2 and len(props) >= 5 and p.area < 0.8 * med:
            partner = _oversplit_partner(labels, p, med, boxes, all_areas)
            if partner:
                issues.append({"code": "half", "partner": partner})
                score += 1.5
        if issues:
            out.append({"label": int(p.label), "centroid": [float(c) for c in p.centroid],
                        "issues": issues, "reason": _reason(issues), "score": round(score, 3)})
    return sorted(out, key=lambda d: -d["score"])


def _oversplit_partner(labels: np.ndarray, prop, med: float, boxes, areas) -> int:
    """A touching neighbour that, together with this object, makes one convex cell of normal size."""
    from skimage.measure import regionprops

    r0, c0, r1, c1 = prop.bbox
    box = (slice(max(r0 - 2, 0), r1 + 2), slice(max(c0 - 2, 0), c1 + 2))
    sub = labels[box]
    ring = ndi.binary_dilation(sub == prop.label, iterations=2) & (sub != prop.label) & (sub > 0)
    for nb in np.unique(sub[ring]):
        if areas[nb] > 0.8 * med or prop.area + areas[nb] > 1.6 * med:
            continue
        nb_box = boxes[nb - 1]
        ub = (slice(min(r0, nb_box[0].start), max(r1, nb_box[0].stop)),
              slice(min(c0, nb_box[1].start), max(c1, nb_box[1].stop)))
        crop = labels[ub]
        union = (crop == prop.label) | (crop == nb)
        if regionprops(union.astype(np.uint8))[0].solidity >= 0.93:
            return int(nb)
    return 0


def find_missed(image: np.ndarray, labels: np.ndarray, min_area: int | None = None) -> list[dict]:
    """Objects visible in the image but not labelled: candidates for cells the detector MISSED.

    Polarity (bright cells on dark = fluorescence, or dark on bright = brightfield) is learned from the
    cells that ARE labelled. Works on single-channel 2-D/3-D images.
    """
    from skimage.filters import gaussian, threshold_otsu

    img = gaussian(image.astype(np.float32), sigma=1.5, preserve_range=True)
    fg_lab = labels > 0
    bright = True
    if fg_lab.any() and (~fg_lab).any():
        bright = img[fg_lab].mean() >= img[~fg_lab].mean()
    if not bright:
        img = img.max() - img
    sample = img[:: max(1, img.shape[0] // 512)]
    try:
        thr = threshold_otsu(sample)
    except ValueError:
        return []
    if min_area is None:
        cell_areas = np.bincount(labels.ravel())[1:]
        cell_areas = cell_areas[cell_areas > 0]
        min_area = int(0.3 * np.median(cell_areas)) if len(cell_areas) else 30
    min_area = max(min_area, 5)
    # bright pixels that are NOT near any label; opening removes thin rims around labelled cells,
    # so touching / crowded cells are found too
    near_label = ndi.binary_dilation(fg_lab, iterations=2) if fg_lab.any() else fg_lab
    free = (img > thr) & ~near_label
    r = max(1, int(round(math.sqrt(min_area / math.pi) * 0.5)))
    free = ndi.binary_opening(free, structure=ndi.generate_binary_structure(img.ndim, 1), iterations=r)
    objects, n = ndi.label(free)
    if n == 0:
        return []
    areas = np.bincount(objects.ravel())
    centroids = ndi.center_of_mass(free, objects, range(1, n + 1))
    out = []
    for i in range(1, n + 1):
        if areas[i] >= min_area:
            issues = [{"code": "missed_bright" if bright else "missed_dark", "area": int(areas[i])}]
            out.append({"label": 0, "centroid": [float(c) for c in centroids[i - 1]], "issues": issues,
                        "reason": _reason(issues), "score": 2 + min(areas[i] / min_area, 3) * 0.1,
                        "area": int(areas[i])})
    return sorted(out, key=lambda d: -d["score"])
