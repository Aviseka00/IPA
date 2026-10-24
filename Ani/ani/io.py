"""Reading and writing images, masks and detector predictions.

Everything here is plain numpy so it can be tested without a GUI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
MASK_EXTS = [".tif", ".tiff", ".png", ".npy", ".txt"]  # search order for predictions
MASK_SUFFIXES = ("", "_mask", "_masks", "_labels", "_cp_masks")


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    try:
        names = list(folder.iterdir())
    except OSError:
        return []
    return sorted(p for p in names if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def read_image(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        img = _read_tiff(path)
    else:
        import imageio.v3 as iio

        img = iio.imread(str(path))
    return np.squeeze(np.asarray(img))


def _read_tiff(path: Path) -> np.ndarray:
    """Read a TIFF fast. Lab LZW RGB scans go through Pillow first; stacks use tifffile."""
    import logging
    import warnings

    # Single-page RGB/LZW scans (the usual Ani case) are faster and more reliable via Pillow.
    try:
        from PIL import Image

        with Image.open(path) as handle:
            frames = int(getattr(handle, "n_frames", 1) or 1)
            if frames == 1:
                handle.load()
                return np.ascontiguousarray(np.asarray(handle))
    except Exception:
        pass

    import tifffile

    log = logging.getLogger("tifffile")
    previous = log.level
    log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.ascontiguousarray(tifffile.imread(str(path)))
    except (ValueError, OSError, RuntimeError):
        from PIL import Image

        with Image.open(path) as handle:
            handle.load()
            return np.ascontiguousarray(np.asarray(handle))
    finally:
        log.setLevel(previous)


def guess_layout(img: np.ndarray, axis0: str = "auto") -> dict:
    """Decide how to show an image.

    Returns {"rgb": bool, "channel_axis": int|None, "spatial_shape": tuple}.
    axis0: "auto", "channels" or "z" (user override for 3-D arrays).
    """
    if img.ndim == 3 and img.shape[-1] in (3, 4) and img.dtype == np.uint8 and axis0 == "auto":
        return {"rgb": True, "channel_axis": None, "spatial_shape": img.shape[:2]}
    if img.ndim >= 3:
        is_channels = axis0 == "channels" or (axis0 == "auto" and img.shape[0] <= 4)
        if is_channels:
            return {"rgb": False, "channel_axis": 0, "spatial_shape": img.shape[1:]}
    return {"rgb": False, "channel_axis": None, "spatial_shape": img.shape}


def contrast_limits(img: np.ndarray) -> tuple[float, float]:
    """Robust auto-contrast (works for 8/16-bit fluorescence and brightfield)."""
    sample = img[:: max(1, img.shape[0] // 512)] if img.ndim >= 2 else img
    lo, hi = np.percentile(sample, (0.5, 99.8))
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def to_label_dtype(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    m = int(mask.max()) if mask.size else 0
    return mask.astype(np.uint16 if m < 65535 else np.uint32, copy=False)


def rgb_mask_to_labels(mask: np.ndarray) -> np.ndarray:
    """Colour-coded instance masks (each cell a colour) -> integer labels."""
    flat = mask[..., :3].reshape(-1, 3)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)
    labels = inverse.reshape(mask.shape[:-1]).astype(np.int64)
    bg = np.where((colours == 0).all(axis=1))[0]
    if len(bg):  # keep black as background 0
        b = bg[0]
        labels = np.where(labels == b, 0, np.where(labels < b, labels + 1, labels))
    else:
        labels = labels + 1
    return labels


def yolo_seg_to_mask(txt_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict[int, int]]:
    """YOLO segmentation txt (cls x1 y1 x2 y2 ... normalised) -> label mask + {label: class_index}."""
    from skimage.draw import polygon as draw_polygon

    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint16)
    classes: dict[int, int] = {}
    label = 0
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        pts = np.array(parts[1:], float).reshape(-1, 2)
        rr, cc = draw_polygon(pts[:, 1] * h, pts[:, 0] * w, shape=(h, w))
        label += 1
        mask[rr, cc] = label
        classes[label] = cls
    return mask, classes


def read_mask(path: Path, shape: tuple | None = None) -> tuple[np.ndarray, dict[int, int]]:
    """Read any supported mask/prediction file. Returns (labels, {label: class_index})."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        return to_label_dtype(np.load(path)), {}
    if ext == ".txt":
        if shape is None:
            raise ValueError("Image shape is needed to read YOLO txt predictions")
        return yolo_seg_to_mask(path, shape)
    m = read_image(path)
    if m.ndim == 3 and m.shape[-1] in (3, 4) and shape is not None and m.shape[:2] == tuple(shape[:2]):
        m = rgb_mask_to_labels(m)
    return to_label_dtype(m), {}


def find_mask(folder: Path, stem: str) -> Path | None:
    folder = Path(folder)
    for suffix in MASK_SUFFIXES:
        for ext in MASK_EXTS:
            p = folder / f"{stem}{suffix}{ext}"
            if p.exists():
                return p
    return None


def write_mask(path: Path, mask: np.ndarray) -> None:
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    tifffile.imwrite(str(tmp), to_label_dtype(mask), compression="zlib")
    tmp.replace(path)  # atomic: a crash never leaves a half-written label file
