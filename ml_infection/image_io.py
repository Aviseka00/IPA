"""Load microscopy images (PNG/JPEG/TIFF) as RGB uint8."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_image_rgb(path: Path) -> np.ndarray:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in {".tif", ".tiff"}:
        try:
            im = Image.open(path).convert("RGB")
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, None)
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.float32)
                arr -= arr.min()
                m = arr.max() + 1e-8
                arr = (arr / m * 255).astype(np.uint8)
        return arr.astype(np.uint8)
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def letterbox(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    pad_value_img: int = 0,
    pad_value_mask: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resize with aspect ratio preserved to fit in size×size, pad to square."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    import cv2

    ri = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out_i = np.full((size, size, 3), pad_value_img, dtype=image.dtype)
    out_m = np.full((size, size), pad_value_mask, dtype=mask.dtype)
    dy = (size - nh) // 2
    dx = (size - nw) // 2
    out_i[dy : dy + nh, dx : dx + nw] = ri
    out_m[dy : dy + nh, dx : dx + nw] = rm
    meta = {"scale": scale, "pad_x": dx, "pad_y": dy, "orig_hw": (h, w)}
    return out_i, out_m, meta


def preprocess_clahe_lab(rgb: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Contrast enhancement on luminance (helps plate scans / faint stains). RGB uint8 in/out."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
