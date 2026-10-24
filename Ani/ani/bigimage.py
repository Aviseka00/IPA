"""Open large scans as an overview so 16 GB PCs never keep a 50k image in RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# Soft limit for a full decode on a ~16 GB machine (RGB uint8 ≈ 3 bytes/px).
MAX_FULL_PIXELS = 24_000_000
MAX_OVERVIEW_SIDE = 4096


def image_pixel_count(image: np.ndarray) -> int:
    image = np.asarray(image)
    if image.ndim < 2:
        return int(image.size)
    return int(image.shape[0]) * int(image.shape[1])


def needs_overview(image: np.ndarray) -> bool:
    return image_pixel_count(image) > MAX_FULL_PIXELS or max(image.shape[:2]) > MAX_OVERVIEW_SIDE * 2


def maybe_overview(image: np.ndarray, max_side: int = MAX_OVERVIEW_SIDE) -> tuple[np.ndarray, dict]:
    """Return ``(display_image, info)``. Large arrays become a downsampled overview."""
    image = np.asarray(image)
    info = {
        "overview": False,
        "scale": 1.0,
        "full_shape": tuple(int(v) for v in image.shape),
        "note": "",
    }
    if image.ndim < 2 or not needs_overview(image):
        return np.ascontiguousarray(image), info
    height, width = int(image.shape[0]), int(image.shape[1])
    factor = max(height / max_side, width / max_side, 1.0)
    if factor <= 1.0:
        return np.ascontiguousarray(image), info
    out_h = max(1, int(round(height / factor)))
    out_w = max(1, int(round(width / factor)))
    zoom = [out_h / height, out_w / width]
    if image.ndim == 3:
        zoom.append(1.0)
    order = 0 if np.issubdtype(image.dtype, np.integer) else 1
    small = ndi.zoom(image, zoom, order=order)
    if image.ndim == 3:
        small = small[:out_h, :out_w, : image.shape[2]]
    else:
        small = small[:out_h, :out_w]
    if np.issubdtype(image.dtype, np.integer):
        small = np.rint(small).astype(image.dtype, copy=False)
    info.update({
        "overview": True,
        "scale": float(factor),
        "full_shape": (height, width) + tuple(image.shape[2:]),
        "note": (
            f"Large image ({width}×{height}). Showing a {out_w}×{out_h} overview "
            "so this PC stays responsive. Annotations apply to this overview."
        ),
    })
    return np.ascontiguousarray(small), info


def read_for_display(path) -> tuple[np.ndarray, dict]:
    """Read a file; shrink huge images before they fill RAM when Pillow can help."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        try:
            from PIL import Image

            with Image.open(path) as handle:
                width, height = handle.size
                pixels = int(width) * int(height)
                if pixels > MAX_FULL_PIXELS or max(width, height) > MAX_OVERVIEW_SIDE * 2:
                    preview = handle.copy()
                    preview.thumbnail((MAX_OVERVIEW_SIDE, MAX_OVERVIEW_SIDE), Image.Resampling.BILINEAR)
                    arr = np.ascontiguousarray(np.asarray(preview.convert("RGB") if preview.mode not in ("L", "RGB", "RGBA") else preview))
                    if arr.ndim == 3 and arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    info = {
                        "overview": True,
                        "scale": max(width / arr.shape[1], height / arr.shape[0], 1.0),
                        "full_shape": (height, width),
                        "note": (
                            f"Large image ({width}×{height}). Showing a {arr.shape[1]}×{arr.shape[0]} overview "
                            "so this PC stays responsive. Annotations apply to this overview."
                        ),
                    }
                    return arr, info
        except Exception:
            pass
    from . import io

    return maybe_overview(io.read_image(path))
