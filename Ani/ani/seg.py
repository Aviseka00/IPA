"""Save and reload one annotated image.

`*_seg.npy` is the file to keep. It holds the instance mask (0 = background)
and the class of every cell. A readable JSON file, a COCO file and a YOLO-seg
file are written beside it for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import io, tools
from .export import mask_to_polygons
from .safety import atomic_write


def prepare_for_display(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return a 2-D or RGB view of `image` plus a short note if a plane was chosen.

    The note is empty when the whole image is shown. `plane` and `channel` are
    0-based indexes when only one slice of a volume or stack is shown.
    """
    img = np.squeeze(np.asarray(image))
    info = {"note": "", "plane": None, "channel": None}
    notes: list[str] = []
    while img.ndim > 3:
        mid = int(img.shape[0] // 2)
        img = np.take(img, mid, axis=0)
        info["plane"] = mid
        notes.append("Showing one plane of a multi-dimensional image.")
    if img.ndim < 2:
        raise ValueError(f"Cannot display an image with shape {tuple(np.asarray(image).shape)}")
    if img.ndim == 3 and img.shape[-1] <= 4 and img.shape[-1] < min(img.shape[0], img.shape[1]):
        channels = int(img.shape[-1])
        if channels in (3, 4) and img.dtype == np.uint8:
            info["note"] = " ".join(notes)
            return np.ascontiguousarray(img[..., :3]), info
        info["channel"] = 0
        notes.append(f"Showing channel 1 of {channels}.")
        info["note"] = " ".join(notes)
        return img[..., 0], info
    if img.ndim == 2:
        info["note"] = " ".join(notes)
        return img, info
    layout = io.guess_layout(img)
    if layout["rgb"]:
        info["note"] = " ".join(notes)
        return np.ascontiguousarray(img[..., :3]), info
    if layout["channel_axis"] == 0:
        count = int(img.shape[0])
        if count in (3, 4) and img.dtype == np.uint8:
            info["note"] = " ".join(notes)
            return np.ascontiguousarray(np.moveaxis(img[:3], 0, -1)), info
        info["channel"] = 0
        notes.append(f"Showing channel 1 of {count}.")
        info["note"] = " ".join(notes)
        return img[0], info
    mid = int(img.shape[0] // 2)
    info["plane"] = mid
    notes.append(f"Showing plane {mid + 1} of {img.shape[0]}.")
    info["note"] = " ".join(notes)
    return img[mid], info


# Shown in the Save dialog, in this order. The label is what the file-type list displays.
SAVE_KINDS = (
    ("npy", "NumPy seg.npy (*.npy)"),
    ("tif", "Label mask TIFF (*.tif *.tiff)"),
    ("png", "Label mask PNG (*.png)"),
    ("coco", "COCO JSON (*.json)"),
    ("yolo", "YOLO segmentation (*.txt)"),
    ("all", "All annotation formats (*.npy *.tif *.png *.json *.txt)"),
)
SAVE_FILTER = ";;".join(label for _, label in SAVE_KINDS)


def kind_from_filter(selected: str) -> str:
    """Map the Save dialog's file-type choice to a format name."""
    text = (selected or "").lower()
    if "all annotation" in text:
        return "all"
    if "yolo" in text or text.endswith("(*.txt)"):
        return "yolo"
    if "coco" in text:
        return "coco"
    if "png" in text:
        return "png"
    if "tif" in text:
        return "tif"
    return "npy"


def annotation_paths(path: Path) -> dict[str, Path]:
    """Paths written for one save. `cells_seg.npy` keeps that name; sidecars use `cells`."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        path = path.with_suffix(".npy")
    stem = path.stem[:-4] if path.stem.endswith("_seg") else path.stem
    folder = path.parent
    return {
        "seg": path,
        "classes": folder / f"{stem}_classes.json",
        "coco": folder / f"{stem}_coco.json",
        "yolo": folder / f"{stem}_yolo.txt",
    }


def write_segmentation(path: Path, masks: np.ndarray, class_names: list[str], label_class: dict[int, str],
                       image_name: str = "", plane: int | None = None, channel: int | None = None,
                       kind: str = "all") -> list[Path]:
    """Write the chosen annotation format. `kind` is npy, tif, png, coco, yolo, or all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = io.to_label_dtype(np.asarray(masks))
    names = [str(n) for n in class_names] or ["cell"]
    present = {int(i) for i in tools.label_ids(masks)}
    classes = {int(i): str(label_class.get(int(i), names[0])) for i in sorted(present)}
    kind = kind if kind in {k for k, _ in SAVE_KINDS} else "all"
    if kind == "all":
        paths = annotation_paths(path if path.suffix.lower() == ".npy" else path.with_suffix(".npy"))
        _write_npy(paths["seg"], masks, names, classes, image_name, plane, channel)
        _write_classes_json(paths["classes"], masks, names, classes, image_name, plane, channel)
        written = [paths["seg"], paths["classes"]]
        if masks.ndim == 2:
            _write_coco(paths["coco"], masks, names, classes, image_name)
            _write_yolo(paths["yolo"], masks, names, classes)
            written.extend((paths["coco"], paths["yolo"]))
        return written
    target = _path_for_kind(path, kind)
    if kind == "npy":
        _write_npy(target, masks, names, classes, image_name, plane, channel)
        return [target]
    if kind in ("tif", "png"):
        _write_mask_image(target, masks)
        classes_path = target.with_name(f"{target.stem}_classes.json")
        _write_classes_json(classes_path, masks, names, classes, image_name, plane, channel)
        return [target, classes_path]
    if kind == "coco":
        _write_coco(target, masks, names, classes, image_name)
        return [target]
    _write_yolo(target, masks, names, classes)
    return [target]


def _plain_stem(path: Path) -> str:
    stem = path.stem
    for tag in ("_seg", "_labels", "_coco", "_yolo", "_classes"):
        if stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def _path_for_kind(path: Path, kind: str) -> Path:
    """Keep a filename whose suffix already matches the chosen type."""
    suffix = path.suffix.lower()
    if kind == "npy" and suffix == ".npy":
        return path
    if kind == "tif" and suffix in {".tif", ".tiff"}:
        return path
    if kind == "png" and suffix == ".png":
        return path
    if kind == "coco" and suffix == ".json":
        return path
    if kind == "yolo" and suffix == ".txt":
        return path
    stem = _plain_stem(path)
    name = {"npy": f"{stem}_seg.npy", "tif": f"{stem}_labels.tif", "png": f"{stem}_labels.png",
            "coco": f"{stem}_coco.json", "yolo": f"{stem}_yolo.txt"}[kind]
    return path.with_name(name)


def _write_npy(path: Path, masks, names, classes, image_name, plane, channel) -> None:
    payload = {
        "format": "ani-seg-1", "image": image_name, "masks": masks, "classes": names,
        "label_class": classes, "plane": plane, "channel": channel,
    }
    atomic_write(path, lambda target: np.save(target, payload))


def _write_classes_json(path: Path, masks, names, classes, image_name, plane, channel) -> None:
    meta = {
        "format": "ani-seg-1",
        "image": image_name,
        "height": int(masks.shape[0]) if masks.ndim >= 2 else 0,
        "width": int(masks.shape[1]) if masks.ndim >= 2 else 0,
        "classes": names,
        "instances": {str(k): v for k, v in classes.items()},
        "plane": plane,
        "channel": channel,
        "note": "Load masks with np.load(path, allow_pickle=True).item()['masks']. "
                "0 is background. instances maps each cell id to a class name.",
    }
    text = json.dumps(meta, indent=2)
    atomic_write(path, lambda target: target.write_text(text, encoding="utf-8"))


def _write_mask_image(path: Path, masks: np.ndarray) -> None:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        atomic_write(path, lambda target: tifffile.imwrite(target, masks))
        return
    import imageio.v3 as iio
    atomic_write(path, lambda target: iio.imwrite(target, masks.astype(np.uint16)))


def read_seg(path: Path) -> tuple[np.ndarray, list[str], dict[int, str]]:
    """Read a seg.npy file. A plain integer array is accepted too."""
    obj = np.load(Path(path), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if isinstance(obj, dict):
        masks = io.to_label_dtype(np.asarray(obj["masks"]))
        names = [str(n) for n in obj.get("classes") or ["cell"]] or ["cell"]
        raw = obj.get("label_class") or {}
        classes = {int(k): str(v) for k, v in dict(raw).items()}
        return masks, names, classes
    return io.to_label_dtype(np.asarray(obj)), ["cell"], {}


def _write_coco(path: Path, masks: np.ndarray, names: list[str], classes: dict[int, str], image_name: str) -> None:
    from scipy import ndimage as ndi

    h, w = masks.shape[:2]
    index = {n: i for i, n in enumerate(names)}
    boxes = ndi.find_objects(masks)
    annotations = []
    for lab in tools.label_ids(masks):
        lab = int(lab)
        box = boxes[lab - 1] if 0 < lab <= len(boxes) else None
        polys = mask_to_polygons(masks, lab, box=box)
        if not polys:
            continue
        ys, xs = np.nonzero(masks[box] == lab if box is not None else masks == lab)
        if box is not None:
            ys, xs = ys + box[0].start, xs + box[1].start
        x0, y0 = float(xs.min()), float(ys.min())
        cidx = index.get(classes.get(lab, names[0]), 0)
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": 1,
            "category_id": cidx + 1,
            "iscrowd": 0,
            "area": int((masks == lab).sum()),
            "bbox": [x0, y0, float(xs.max() - x0 + 1), float(ys.max() - y0 + 1)],
            "segmentation": [[round(float(v), 2) for v in p.ravel()] for p in polys],
        })
    coco = {
        "images": [{"id": 1, "file_name": image_name or "image", "height": h, "width": w}],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
        "annotations": annotations,
    }
    text = json.dumps(coco)
    atomic_write(path, lambda target: target.write_text(text, encoding="utf-8"))


def mask_from_glide(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Turn a freehand stroke into a filled cell.

    The stroke is smoothed, closed, and drawn so every pixel of the boundary
    is covered. The inside of that boundary is filled.
    """
    from scipy import ndimage as ndi
    from skimage.draw import line

    pts = _stroke_points(points)
    if len(pts) < 4:
        return None
    curve = _densify_closed(_smooth_stroke(pts), step=0.5)
    h, w = int(shape[0]), int(shape[1])
    boundary = np.zeros((h, w), bool)
    n = len(curve)
    for i in range(n):
        a, b = curve[i], curve[(i + 1) % n]
        rr, cc = line(int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1])))
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if ok.any():
            boundary[rr[ok], cc[ok]] = True
    if not boundary.any():
        return None
    # One pixel around the stroke, so the glide covers the edge it follows.
    band = ndi.binary_dilation(boundary, iterations=1)
    return ndi.binary_fill_holes(band)


def _stroke_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 2), float)
    pts = pts[:, :2]
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) >= 1:
            keep.append(i)
    return pts[keep]


def _smooth_stroke(pts: np.ndarray) -> np.ndarray:
    """Take the jitter out of a stroke without pulling it off the cursor.

    The ends stay where they were drawn. Wrapping the smooth onto a stroke
    that has not been closed yet drags the start and end toward each other.
    """
    from scipy.ndimage import gaussian_filter1d

    if len(pts) < 6:
        return pts
    sigma = min(0.8, max(0.35, len(pts) / 80))
    return np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma, mode="nearest"),
        gaussian_filter1d(pts[:, 1], sigma, mode="nearest"),
    ])


def _densify_closed(pts: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*(np.diff(closed, axis=0).T))
    total = float(seg.sum())
    if total < 1:
        return pts
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0, dist[-1], max(int(total / step), len(pts)), endpoint=False)
    return np.column_stack([np.interp(target, dist, closed[:, 0]), np.interp(target, dist, closed[:, 1])])


def _write_yolo(path: Path, masks: np.ndarray, names: list[str], classes: dict[int, str]) -> None:
    from scipy import ndimage as ndi

    h, w = masks.shape[:2]
    index = {n: i for i, n in enumerate(names)}
    boxes = ndi.find_objects(masks)
    lines = []
    for lab in tools.label_ids(masks):
        lab = int(lab)
        box = boxes[lab - 1] if 0 < lab <= len(boxes) else None
        polys = mask_to_polygons(masks, lab, box=box)
        if not polys:
            continue
        cidx = index.get(classes.get(lab, names[0]), 0)
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in polys[0])
        lines.append(f"{cidx} {coords}")
    text = "\n".join(lines) + ("\n" if lines else "")
    atomic_write(path, lambda target: target.write_text(text, encoding="utf-8"))
