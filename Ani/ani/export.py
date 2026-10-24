"""Export corrected labels as a training dataset: instance masks, COCO JSON, YOLO-seg."""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from . import io, tools
from .project import Project


def mask_to_polygons(labels: np.ndarray, label: int, tolerance: float = 1.0, box=None) -> list[np.ndarray]:
    """Outer contour(s) of one object as (N, 2) arrays of (x, y), biggest first.
    `box` (slices from ndimage.find_objects) avoids scanning the whole image."""
    from skimage.measure import approximate_polygon, find_contours

    box = box or tools._bbox(labels == label, pad=1)
    from scipy import ndimage as ndi

    # outer boundaries only: COCO/YOLO polygons cannot express holes, an inner contour would be read as
    # an extra filled piece of the object
    region = np.pad(ndi.binary_fill_holes(labels[box] == label), 1)
    off = np.array([box[1].start, box[0].start], float) - 1  # (x, y) offset of the crop, minus padding
    polys = []
    for c in find_contours(region.astype(np.uint8), 0.5):
        c = approximate_polygon(c, tolerance)
        if len(c) >= 3:
            polys.append(np.clip(c[:, ::-1] + off, 0, None))
    polys.sort(key=len, reverse=True)
    return polys


def _norm8(ch: np.ndarray) -> np.ndarray:
    lo, hi = io.contrast_limits(ch)
    return np.clip((ch.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def to_uint8(img: np.ndarray, layout: dict) -> np.ndarray:
    """8-bit grey or RGB version of any image (most detection trainers can't read 16-bit/multi-channel TIFF)."""
    if layout["rgb"]:
        return img[..., :3].astype(np.uint8)
    if layout["channel_axis"] is not None:
        chans = [_norm8(img[i]) for i in range(min(3, img.shape[0]))]
        if len(chans) == 1:
            return chans[0]
        while len(chans) < 3:
            chans.append(np.zeros_like(chans[0]))
        return np.stack(chans, -1)
    return _norm8(img)


def export_dataset(project: Project, out_dir: Path | None = None, formats=("masks", "coco", "yolo"),
                   only_done: bool = True, val_fraction: float = 0.2, seed: int = 0,
                   convert_8bit: bool = True, progress=None) -> Path:
    out_dir = Path(out_dir or project.root / "exports" / time.strftime("dataset-%Y%m%d-%H%M%S"))
    images = [p for p in project.images() if project.has_labels(p.stem)
              and (not only_done or project.status(p.stem) == "done")]
    if not images:
        raise ValueError("No images to export. Save labels (and mark them Done, or untick 'only Done').")
    rng = random.Random(seed)
    order = images[:]
    rng.shuffle(order)
    n_val = int(round(len(order) * val_fraction)) if len(order) > 1 else 0
    split_of = {p.stem: ("val" if i < n_val else "train") for i, p in enumerate(order)}

    cats = project.categories()
    cat_index = {(s, c): i for i, (s, c, _) in enumerate(cats)}
    coco = {s: {"images": [], "annotations": [],
                "categories": [{"id": i + 1, "name": n, "supercategory": s} for i, (s, _, n) in enumerate(cats)]}
            for s in ("train", "val")}
    ann_id, skipped_3d, counts = 1, [], {}

    for n, img_path in enumerate(images):
        stem, split = img_path.stem, split_of[img_path.stem]
        if progress:
            progress(n, len(images), stem)
        image = io.read_image(img_path)
        layout = io.guess_layout(image)
        shape = tuple(layout["spatial_shape"])
        sets = {s: project.load_labels(s, stem, shape)[:2] for s in project.label_sets
                if project.label_path(s, stem).exists()}

        if "masks" in formats:
            d = out_dir / "masks" / split
            (d / "images").mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, d / "images" / img_path.name)
            for s, (labels, cls_map) in sets.items():
                io.write_mask(d / "labels" / s / f"{stem}.tif", labels)
                (d / "labels" / s / f"{stem}.classes.json").write_text(
                    json.dumps({str(k): v for k, v in cls_map.items()}))

        if len(shape) != 2:  # COCO / YOLO are 2-D formats
            skipped_3d.append(stem)
            continue
        h, w = shape
        out_name = f"{stem}.png" if convert_8bit else img_path.name
        for fmt in ("coco", "yolo"):
            if fmt in formats:
                d = out_dir / fmt / "images" / split
                d.mkdir(parents=True, exist_ok=True)
                if convert_8bit:
                    import imageio.v3 as iio
                    iio.imwrite(d / out_name, to_uint8(image, layout))
                else:
                    shutil.copy2(img_path, d / out_name)
        image_id = len(coco[split]["images"]) + 1
        coco[split]["images"].append({"id": image_id, "file_name": out_name, "height": h, "width": w})
        yolo_lines = []
        for s, (labels, cls_map) in sets.items():
            default = project.classes(s)[0]
            from scipy import ndimage as ndi
            boxes = ndi.find_objects(labels)
            for lab in tools.label_ids(labels):
                box = boxes[int(lab) - 1]
                polys = mask_to_polygons(labels, int(lab), box=box)
                if not polys:
                    continue
                cname = cls_map.get(int(lab), default)
                cidx = cat_index.get((s, cname), cat_index[(s, default)])
                counts[cats[cidx][2]] = counts.get(cats[cidx][2], 0) + 1
                ys, xs = np.nonzero(labels[box] == lab)
                ys, xs = ys + box[0].start, xs + box[1].start
                x0, y0 = float(xs.min()), float(ys.min())
                coco[split]["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": cidx + 1, "iscrowd": 0,
                    "area": int(len(xs)),
                    "bbox": [x0, y0, float(xs.max() - x0 + 1), float(ys.max() - y0 + 1)],
                    "segmentation": [[round(float(v), 2) for v in p.ravel()] for p in polys],
                })
                ann_id += 1
                yolo_lines.append(f"{cidx} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in polys[0]))
        if "yolo" in formats:
            d = out_dir / "yolo" / "labels" / split
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{stem}.txt").write_text("\n".join(yolo_lines))

    if "coco" in formats:
        for split, data in coco.items():
            if data["images"]:
                (out_dir / "coco").mkdir(parents=True, exist_ok=True)
                (out_dir / "coco" / f"{split}.json").write_text(json.dumps(data))
    if "yolo" in formats and (out_dir / "yolo").exists():
        names = "\n".join(f"  {i}: {n}" for i, (_, _, n) in enumerate(cats))
        (out_dir / "yolo" / "data.yaml").write_text(
            f"path: {(out_dir / 'yolo').resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    (out_dir / "README.txt").write_text(
        f"Exported by Ani on {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"Images: {len(images)} (validation: {n_val})\n"
        f"Label sets: {', '.join(project.label_sets)}\n"
        f"Objects per class: {json.dumps(counts)}\n"
        f"COCO/YOLO images: {'8-bit PNG (auto-contrast)' if convert_8bit else 'original files'}\n"
        + (f"3-D images exported only as masks: {', '.join(skipped_3d)}\n" if skipped_3d else ""))
    if progress:
        progress(len(images), len(images), "done")
    return out_dir
