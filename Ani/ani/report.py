"""One-click densitogram reports: CSV table + profile PNG (no matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .density import DensityResult, analyze_lane, write_density_csv
from .lanes import find_gel_rois, find_lanes, lanes_as_rois
from .safety import atomic_write


def write_profile_png(path, result: DensityResult, title: str = "") -> Path:
    """Draw the densitogram profile with Pillow and save it."""
    path = Path(path)
    width, height = 720, 280
    image = Image.new("RGB", (width, height), (23, 26, 33))
    draw = ImageDraw.Draw(image)
    profile = np.asarray(result.profile, np.float64)
    if profile.size < 2:
        draw.text((24, height // 2), "No profile", fill=(167, 176, 190))
        atomic_write(path, lambda target: image.save(target, format="PNG"))
        return path
    left, top, right, bottom = 48, 28, width - 16, height - 36
    ymax = max(float(profile.max()) * 1.08, 1e-6)
    span = max(profile.size - 1, 1)

    def x_of(sample: float) -> float:
        return left + (sample / span) * (right - left)

    def y_of(value: float) -> float:
        return bottom - (float(value) / ymax) * (bottom - top)

    draw.line([(left, bottom), (right, bottom)], fill=(60, 68, 84), width=1)
    draw.line([(left, top), (left, bottom)], fill=(60, 68, 84), width=1)
    colors = [(255, 92, 92), (232, 107, 255), (255, 176, 32), (61, 220, 151), (122, 162, 255)]
    for peak in result.peaks:
        color = colors[(peak.number - 1) % len(colors)]
        poly = []
        for sample in range(peak.left, peak.right + 1):
            poly.append((x_of(sample), y_of(profile[sample])))
        for sample in range(peak.right, peak.left - 1, -1):
            base = result.baseline[sample]
            level = 0.0 if base != base else float(base)
            poly.append((x_of(sample), y_of(level)))
        if len(poly) >= 3:
            draw.polygon(poly, outline=color)
            draw.line(poly + [poly[0]], fill=color, width=1)
        draw.text((x_of(peak.sample) - 4, y_of(profile[peak.sample]) - 14), str(peak.number), fill=color)
    points = [(x_of(i), y_of(profile[i])) for i in range(profile.size)]
    draw.line(points, fill=(244, 246, 248), width=2)
    if title:
        draw.text((left, 8), title[:64], fill=(244, 246, 248))
    draw.text((4, top), "Density", fill=(167, 176, 190))
    draw.text((left, height - 22), "Position", fill=(167, 176, 190))
    atomic_write(path, lambda target: image.save(target, format="PNG"))
    return path


def densitogram_report(
    image: np.ndarray,
    out_dir,
    stem: str,
    rois: list[dict] | None = None,
    along: str = "auto",
    dark_bands: bool = True,
    notch_percent: float = 8.0,
) -> list[Path]:
    """Write ``{stem}_density.csv`` and one profile PNG per ROI. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rois:
        rois = lanes_as_rois(find_gel_rois(image) or find_lanes(image))
    if not rois:
        return []
    results = [
        analyze_lane(
            image,
            roi=item["roi"],
            along=along,
            dark_bands=dark_bands,
            notch_percent=notch_percent,
            mask=item.get("mask"),
        )
        for item in rois
    ]
    names = [item.get("name") or f"roi {i}" for i, item in enumerate(rois, start=1)]
    written: list[Path] = []
    csv_path = write_density_csv(out_dir / f"{stem}_density.csv", results, names)
    written.append(csv_path)
    for number, (name, result) in enumerate(zip(names, results), start=1):
        png = write_profile_png(out_dir / f"{stem}_roi{number}_profile.png", result, title=str(name))
        written.append(png)
    return written


def batch_densitogram_reports(
    folder,
    out_dir=None,
    along: str = "auto",
    dark_bands: bool = True,
    notch_percent: float = 8.0,
    progress=None,
) -> list[Path]:
    """Run a densitogram report for every image in a folder (auto-lanes/bands when needed)."""
    from . import io
    from .seg import read_seg

    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir is not None else folder / "ani_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = io.list_images(folder)
    written: list[Path] = []
    for index, path in enumerate(images, start=1):
        if progress is not None:
            progress(index, len(images), path.name)
        try:
            image = io.read_image(path)
            from .bigimage import maybe_overview

            image, _info = maybe_overview(image)
            rois = None
            seg = path.with_name(f"{path.stem}_seg.npy")
            if seg.is_file():
                try:
                    masks, _names, _classes = read_seg(seg)
                    rois = _rois_from_mask(masks)
                except Exception:
                    rois = None
            if not rois:
                rois = lanes_as_rois(find_gel_rois(image) or find_lanes(image))
            if not rois:
                continue
            written.extend(densitogram_report(
                image, out_dir / path.stem, path.stem, rois=rois,
                along=along, dark_bands=dark_bands, notch_percent=notch_percent,
            ))
        except Exception:
            continue
    return written


def _rois_from_mask(masks: np.ndarray) -> list[dict]:
    from scipy import ndimage as ndi

    from . import tools

    rois = []
    boxes = ndi.find_objects(masks)
    for lab in tools.label_ids(masks):
        lab = int(lab)
        box = boxes[lab - 1] if 0 < lab <= len(boxes) else None
        if box is None:
            continue
        r0, r1 = int(box[0].start), int(box[0].stop)
        c0, c1 = int(box[1].start), int(box[1].stop)
        if (r1 - r0) < 4 or (c1 - c0) < 4:
            continue
        mask = masks == lab
        rois.append({"kind": "cell", "name": f"cell {lab}", "roi": (r0, c0, r1, c1), "mask": mask, "label": lab})
    return rois
