"""Team-light sharing: package folder copies, COCO, and Label Studio JSON."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .safety import atomic_write
from .seg import write_segmentation


def share_package(
    dest_folder,
    image_path,
    masks: np.ndarray,
    class_names: list[str],
    classes: dict[int, str],
    extras: list[Path] | None = None,
) -> Path:
    """Copy a complete Ani package into ``dest_folder / <stem>/`` for a shared drive."""
    image_path = Path(image_path)
    dest_folder = Path(dest_folder)
    bundle = dest_folder / image_path.stem
    bundle.mkdir(parents=True, exist_ok=True)
    # Image (copy, do not move).
    if image_path.is_file():
        shutil.copy2(image_path, bundle / image_path.name)
    written = write_segmentation(
        bundle / f"{image_path.stem}_seg.npy",
        masks,
        class_names,
        classes,
        image_name=image_path.name,
        kind="all",
    )
    for extra in extras or []:
        extra = Path(extra)
        if extra.is_file():
            shutil.copy2(extra, bundle / extra.name)
    manifest = {
        "format": "ani-share-1",
        "image": image_path.name,
        "files": [p.name for p in written] + [Path(p).name for p in (extras or []) if Path(p).is_file()],
        "note": "Open the image in Ani, or import *_coco.json / label_studio.json elsewhere.",
    }
    text = json.dumps(manifest, indent=2)
    atomic_write(bundle / "ani_share.json", lambda target: target.write_text(text, encoding="utf-8"))
    return bundle


def write_label_studio(
    path,
    masks: np.ndarray,
    class_names: list[str],
    classes: dict[int, str],
    image_name: str = "",
) -> Path:
    """Write a Label Studio tasks JSON with polygon labels (importable offline)."""
    from scipy import ndimage as ndi

    from . import tools
    from .export import mask_to_polygons

    path = Path(path)
    masks = np.asarray(masks)
    height, width = masks.shape[:2]
    names = [str(n) for n in class_names] or ["cell"]
    boxes = ndi.find_objects(masks)
    results = []
    for lab in tools.label_ids(masks):
        lab = int(lab)
        box = boxes[lab - 1] if 0 < lab <= len(boxes) else None
        polys = mask_to_polygons(masks, lab, box=box)
        if not polys:
            continue
        label = str(classes.get(lab, names[0]))
        for poly in polys:
            # Label Studio expects percent coordinates for polygonlabels.
            points = []
            flat = np.asarray(poly, np.float64).reshape(-1, 2)
            for row, col in flat:
                points.append([
                    100.0 * float(col) / max(width, 1),
                    100.0 * float(row) / max(height, 1),
                ])
            results.append({
                "original_width": int(width),
                "original_height": int(height),
                "image_rotation": 0,
                "value": {"points": points, "polygonlabels": [label]},
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "origin": "manual",
            })
    task = {
        "data": {"image": image_name or "image"},
        "annotations": [{"result": results}],
    }
    text = json.dumps([task], indent=2)
    atomic_write(path, lambda target: target.write_text(text, encoding="utf-8"))
    return path
