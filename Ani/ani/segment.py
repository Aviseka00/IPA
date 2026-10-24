"""Whole-image and one-click segmentation for the simple annotator.

Classic always runs. Cellpose is offered only when that package is already installed.
"""
from __future__ import annotations

import numpy as np

from .ai.cellpose_backend import CellposeBackend
from .ai.classic import ClassicBackend


def segment_choices() -> list[tuple[str, str]]:
    """(key, label) pairs for the segment menu. Classic is always first."""
    choices = [("classic", "Classic"), ("spots", "Infected spots")]
    if CellposeBackend("cyto3").available()[0]:
        choices.append(("cellpose:cyto3", "Cellpose cells"))
        choices.append(("cellpose:nuclei", "Cellpose nuclei"))
    return choices


def backend_for_key(key: str):
    if key == "classic":
        return ClassicBackend()
    if key == "cellpose:cyto3":
        return CellposeBackend("cyto3")
    if key == "cellpose:nuclei":
        return CellposeBackend("nuclei")
    raise ValueError(f"Unknown segmenter: {key}")


def to_gray(image: np.ndarray) -> np.ndarray:
    """2-D image for the segmenters. Colour images become a brightness image."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3]
        if rgb.dtype == np.uint8:
            # Integer luma is much faster than float RGB on large scans.
            return (
                (rgb[..., 0].astype(np.uint16) * 54
                 + rgb[..., 1].astype(np.uint16) * 183
                 + rgb[..., 2].astype(np.uint16) * 19) >> 8
            ).astype(np.float32)
        rgb = rgb.astype(np.float32)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError("Segmentation needs a 2-D image.")


def run_segment(image: np.ndarray, key: str) -> np.ndarray:
    """Label every object. The returned array matches the image height and width."""
    if key == "spots":
        from .spots import find_infected

        labels, _classes, _meta = find_infected(image)
        return np.asarray(labels)
    labels = backend_for_key(key).auto_segment(to_gray(image))
    return np.asarray(labels)


def run_click(image: np.ndarray, labels: np.ndarray, point, key: str):
    """Label only the object under the click. Returns (mask, box) or (None, None)."""
    point = tuple(int(round(float(p))) for p in point)
    if key == "spots":
        from .spots import find_infected

        found, _classes, _meta = find_infected(image)
        value = int(found[point]) if found.shape == np.asarray(labels).shape else 0
        if value == 0:
            return None, None
        mask_full = found == value
        rows, cols = np.nonzero(mask_full)
        box = (
            slice(int(rows.min()), int(rows.max()) + 1),
            slice(int(cols.min()), int(cols.max()) + 1),
        )
        return mask_full[box], box
    return backend_for_key(key).segment_at(to_gray(image), point, np.asarray(labels))
