"""Measurements and crops for every cell the user outlined.

The outlines stay as they were drawn. A colour image also gets a nucleus estimated
inside each cell from the haematoxylin channel, and cytoplasm is the rest of the cell.
The CSV is one row per cell. Patches and masks are enough to put that cell back.
"""
from __future__ import annotations

import csv
import json
import zlib
from math import factorial
from pathlib import Path

import numpy as np

from .safety import atomic_write, feature_job_error

PAD = 10
CNN_SIZE = 128
CHANNELS = ("gray", "red", "green", "blue", "hue", "sat", "val", "l", "a", "b", "hem", "eos", "dab")
REGIONS = ("cell", "core", "nucleus", "cytoplasm")
INTENSITY_STATS = (
    "mean", "std", "min", "max", "median", "p5", "p25", "p75", "p95", "skew", "kurtosis", "integrated",
)
SHAPE_NAMES = (
    "area", "perimeter", "equivalent_diameter", "major_axis", "minor_axis", "aspect_ratio",
    "eccentricity", "orientation_deg", "solidity", "extent", "circularity", "convex_area",
    "feret_max", "feret_min", "convexity_defects", "convexity_depth",
)
ZERNIKE_ORDERS = tuple((n, m) for n in range(5) for m in range(n + 1) if (n - m) % 2 == 0)
_SHAPE_EXTRA = (
    tuple(f"hu_{i}" for i in range(1, 8))
    + tuple(f"zernike_n{n}_m{m}" for n, m in ZERNIKE_ORDERS)
    + tuple(f"efd_{i}" for i in range(1, 11))
)
_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
_CLASS_COLORS = (
    (220, 80, 70), (64, 176, 112), (232, 176, 64), (90, 140, 230),
    (196, 120, 220), (230, 130, 70),
)


def _names():
    cols = [
        "image", "well", "slide", "magnification", "pixel_size_um", "label", "class",
        "image_height", "image_width", "touches_border", "neighbour_count",
        "bbox_row", "bbox_col", "bbox_height", "bbox_width",
        "centroid_row", "centroid_col", "pad",
        "crop_row", "crop_col", "crop_height", "crop_width",
        "contour_json", "rle", "rle_height", "rle_width",
        "patch", "mask_cell", "mask_nucleus", "mask_cytoplasm", "cnn_patch",
        "nucleus_count", "nucleus_cell_area_ratio", "nucleus_offset_row", "nucleus_offset_col",
        "gray_corrected_mean", "stain_positive_fraction",
        "boundary_inside_mean", "boundary_outside_mean", "boundary_contrast",
        "bright_puncta_count", "bright_puncta_area", "dark_puncta_count", "dark_puncta_area",
    ]
    cols.extend(SHAPE_NAMES)
    cols.append("feret_diameter")
    cols.extend((
        "area_um2", "perimeter_um", "equivalent_diameter_um",
        "major_axis_um", "minor_axis_um", "feret_max_um", "feret_min_um",
    ))
    cols.extend(_SHAPE_EXTRA)
    cols.extend(f"nuc_{name}" for name in SHAPE_NAMES + _SHAPE_EXTRA)
    cols.extend(("intensity_mean", "intensity_min", "intensity_max", "intensity_std",
                 "intensity_median", "integrated_intensity",
                 "mean_red", "mean_green", "mean_blue", "std_red", "std_green", "std_blue"))
    for region in REGIONS:
        for channel in CHANNELS:
            cols.extend(f"{region}_{channel}_{stat}" for stat in INTENSITY_STATS)
    for channel in ("dab", "hem", "l"):
        cols.extend(f"{channel}_hist_{i:02d}" for i in range(16))
    for channel in ("gray", "dab"):
        for dist in (1, 2, 3):
            cols.extend(f"glcm_{channel}_d{dist}_{prop}" for prop in _GLCM_PROPS)
        cols.extend(f"haralick_{channel}_{i:02d}" for i in range(1, 14))
        cols.extend(f"lbp_{channel}_{i:02d}" for i in range(10))
        for angle in (0, 45, 90, 135):
            for scale in (1, 2):
                cols.append(f"gabor_{channel}_a{angle}_s{scale}")
    return tuple(cols)


FEATURE_COLUMNS = _names()


def extract_cell_features(image: np.ndarray, labels: np.ndarray, classes: dict | None = None,
                          image_name: str = "", well: str = "", slide: str = "",
                          magnification: str = "", pixel_size_um="", stain_matrix=None) -> list[dict]:
    """Measure every outlined cell. `classes` maps a label id to a name such as infected."""
    rows, _ = _measure(
        image, labels, classes, image_name, well, slide, magnification, pixel_size_um, stain_matrix,
    )
    return rows


def write_feature_table(path, rows: list[dict]) -> Path:
    """Write one CSV row per cell."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    def write(target: Path) -> None:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _cell(row.get(key, "")) for key in FEATURE_COLUMNS})

    return atomic_write(path, write)


def export_training_set(out_dir, image, labels, classes=None, image_name="", well="", slide="",
                        magnification="", pixel_size_um="", stain_matrix=None, progress=None) -> Path:
    """Write features.csv, patches, masks, CNN crops, a QC image and a column guide."""
    image = np.asarray(image)
    labels = np.asarray(labels)
    height, width = int(labels.shape[0]), int(labels.shape[1])
    n_cells = len([i for i in np.unique(labels) if int(i) > 0])
    reason = feature_job_error(height, width, n_cells)
    if reason:
        raise RuntimeError(reason)
    out_dir = Path(out_dir)
    if out_dir.suffix.lower() == ".csv":
        csv_path = out_dir
        out_dir = out_dir.parent
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "features.csv"
    rows, artifacts = _measure(
        image, labels, classes, image_name, well, slide, magnification, pixel_size_um, stain_matrix, progress,
    )
    stem = _safe_stem(image_name)
    patches = out_dir / "patches"
    masks = out_dir / "masks"
    cnn = out_dir / "cnn"
    qc_dir = out_dir / "qc"
    for folder in (patches, masks, cnn, qc_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        lab = int(row["label"])
        saved = artifacts["crops"][lab]
        token = f"{stem}_{lab}"
        rel = {
            "patch": patches / f"{token}.png",
            "mask_cell": masks / f"{token}_cell.png",
            "mask_nucleus": masks / f"{token}_nuc.png",
            "mask_cytoplasm": masks / f"{token}_cyto.png",
            "cnn_patch": cnn / f"{token}.png",
        }
        _write_png(rel["patch"], saved["patch"])
        _write_png(rel["mask_cell"], _mask_png(saved["cell"]))
        _write_png(rel["mask_nucleus"], _mask_png(saved["nuc"]))
        _write_png(rel["mask_cytoplasm"], _mask_png(saved["cyto"]))
        _write_png(rel["cnn_patch"], saved["cnn"])
        for key, file in rel.items():
            row[key] = file.relative_to(out_dir).as_posix()
    _write_qc(qc_dir / f"{stem}_qc.png", artifacts["rgb"], labels, artifacts["nuclei"], classes or {})
    write_feature_table(csv_path, rows)
    _write_readme(out_dir / "README.txt")
    return csv_path


def reconstruct_cell(folder, label: int) -> np.ndarray:
    """Rebuild one cell from its saved patch and draw the cell and nucleus contours."""
    folder = Path(folder)
    row = _row_for(folder, label)
    patch = _read_png(folder / row["patch"])
    if patch.ndim == 2:
        patch = np.stack([patch, patch, patch], axis=-1)
    canvas = np.array(patch, copy=True)
    cell = _read_png(folder / row["mask_cell"]) > 0
    nuc = _read_png(folder / row["mask_nucleus"]) > 0
    _paint_contour(canvas, cell, (255, 210, 40))
    _paint_contour(canvas, nuc, (40, 210, 255))
    return canvas


def paste_cells(folder, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Paste every cell onto a blank canvas at its original position, coloured by class."""
    folder = Path(folder)
    rows = _read_rows(folder)
    if shape is None:
        height = max(int(float(row["image_height"])) for row in rows)
        width = max(int(float(row["image_width"])) for row in rows)
        shape = (height, width)
    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)
    for row in rows:
        mask = _read_png(folder / row["mask_cell"]) > 0
        color = _class_color(row.get("class", ""))
        r0, c0 = int(float(row["crop_row"])), int(float(row["crop_col"]))
        r1, c1 = min(shape[0], r0 + mask.shape[0]), min(shape[1], c0 + mask.shape[1])
        view = canvas[r0:r1, c0:c1]
        part = mask[:r1 - r0, :c1 - c0]
        view[part] = color
    return canvas


def mask_from_rle(encoded, height: int, width: int) -> np.ndarray:
    """Decode the CSV backup mask. `encoded` is a JSON list of start, length pairs."""
    pairs = json.loads(encoded) if isinstance(encoded, str) else list(encoded)
    flat = np.zeros(int(height) * int(width), np.uint8)
    for index in range(0, len(pairs), 2):
        start, length = int(pairs[index]), int(pairs[index + 1])
        flat[start:start + length] = 1
    return flat.reshape((int(height), int(width)))


def _measure(image, labels, classes, image_name, well, slide, magnification, pixel_size_um, stain_matrix, progress=None):
    labels = np.asarray(labels)
    channels, corrected = _channels(image, stain_matrix)
    gray = channels["gray"]
    if gray.shape != labels.shape:
        raise ValueError(f"Image is {gray.shape} and the labels are {labels.shape}.")
    rgb = _display_rgb(image)
    classes = {int(k): v for k, v in (classes or {}).items()}
    ids = [int(i) for i in np.unique(labels) if int(i) > 0]
    if progress is not None:
        progress(0, len(ids))
    nuclei = np.zeros(labels.shape, np.uint16)
    crops = {}
    pending = []
    height, width = labels.shape
    for index, lab in enumerate(ids):
        cell = labels == lab
        nuc = _nucleus_mask(channels.get("hem"), _original_rgb(image), cell)
        cyto = cell & ~nuc
        nuclei[nuc] = lab
        shape, contour = _shape_dict(cell)
        nuc_shape, _ = _shape_dict(nuc) if nuc.any() else (_empty_shape(), np.zeros((0, 2)))
        r0, c0, r1, c1 = _bounds(cell)
        r0p, c0p = max(0, r0 - PAD), max(0, c0 - PAD)
        r1p, c1p = min(height, r1 + PAD), min(width, c1 + PAD)
        crop_cell = cell[r0p:r1p, c0p:c1p]
        crop_nuc = nuc[r0p:r1p, c0p:c1p]
        tight = cell[r0:r1, c0:c1]
        core = _eroded(cell, 3)
        regions = {"cell": cell, "core": core if core.any() else cell, "nucleus": nuc, "cytoplasm": cyto}
        row = {key: "" for key in FEATURE_COLUMNS}
        row.update({
            "image": image_name, "well": well, "slide": slide, "magnification": magnification,
            "label": lab, "class": classes.get(lab, ""),
            "image_height": height, "image_width": width,
            "touches_border": int(r0 == 0 or c0 == 0 or r1 == height or c1 == width),
            "bbox_row": r0, "bbox_col": c0, "bbox_height": r1 - r0, "bbox_width": c1 - c0,
            "centroid_row": shape["centroid_row"] + r0, "centroid_col": shape["centroid_col"] + c0,
            "pad": PAD, "crop_row": r0p, "crop_col": c0p,
            "crop_height": r1p - r0p, "crop_width": c1p - c0p,
            "contour_json": json.dumps(_contour_xy(contour)),
            "rle": json.dumps(_rle(tight)), "rle_height": r1 - r0, "rle_width": c1 - c0,
            "nucleus_count": int(nuc.any() and _object_count(nuc)),
            "gray_corrected_mean": float(corrected[cell].mean()) if cell.any() else "",
        })
        if nuc.any():
            total = float(nuc.sum())
            ys, xs = np.nonzero(nuc)
            row["nucleus_cell_area_ratio"] = total / max(float(cell.sum()), 1.0)
            row["nucleus_offset_row"] = float(ys.mean() - row["centroid_row"])
            row["nucleus_offset_col"] = float(xs.mean() - row["centroid_col"])
        else:
            row["nucleus_count"] = 0
        for name in SHAPE_NAMES + _SHAPE_EXTRA:
            row[name] = shape.get(name, "")
        row["feret_diameter"] = shape.get("feret_max", "")
        _apply_pixel_size(row, pixel_size_um)
        if nuc.any():
            for name in SHAPE_NAMES + _SHAPE_EXTRA:
                row[f"nuc_{name}"] = nuc_shape.get(name, "")
        _fill_intensity(row, channels, regions)
        _fill_histograms(row, channels, cell)
        row["stain_positive_fraction"] = _positive_fraction(channels.get("dab"), cell)
        _fill_boundary(row, gray, cell)
        _fill_puncta(row, gray, cell)
        _fill_texture(row, "gray", gray, cell)
        if channels.get("dab") is not None:
            _fill_texture(row, "dab", channels["dab"], cell)
        row["_diameter"] = shape.get("equivalent_diameter", 0.0) or 0.0
        pending.append(row)
        crops[lab] = {
            "patch": rgb[r0p:r1p, c0p:c1p],
            "cell": crop_cell,
            "nuc": crop_nuc,
            "cyto": cyto[r0p:r1p, c0p:c1p],
            "cnn": _letterbox(rgb[r0p:r1p, c0p:c1p], CNN_SIZE),
        }
        if progress is not None:
            progress(index + 1, len(ids))
    _fill_neighbours(pending)
    for row in pending:
        row.pop("_diameter", None)
    return pending, {"crops": crops, "nuclei": nuclei, "rgb": rgb}


def _fill_neighbours(rows):
    centers = [(row, float(row["centroid_row"]), float(row["centroid_col"]), float(row["_diameter"] or 0))
               for row in rows]
    for row, y, x, _ in centers:
        count = 0
        reach = 2 * max(float(row["_diameter"] or 0), 1.0)
        for other, oy, ox, _ in centers:
            if other is row:
                continue
            if np.hypot(y - oy, x - ox) <= reach:
                count += 1
        row["neighbour_count"] = count


def _channels(image, stain_matrix):
    gray, rgb = _split_image(image)
    out = {name: None for name in CHANNELS}
    out["gray"] = gray
    corrected = _background_corrected(gray)
    if rgb is None:
        return out, corrected
    out["red"], out["green"], out["blue"] = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rgb01 = rgb.astype(np.float64)
    peak = float(np.max(rgb01)) if rgb01.size else 1.0
    if peak > 1:
        rgb01 = rgb01 / (255.0 if peak <= 255 else peak)
    from skimage.color import rgb2hed, rgb2hsv, rgb2lab, separate_stains
    hsv = rgb2hsv(np.clip(rgb01, 0, 1))
    lab = rgb2lab(np.clip(rgb01, 0, 1))
    hed = rgb2hed(np.clip(rgb01, 0, 1)) if stain_matrix is None else separate_stains(np.clip(rgb01, 0, 1), stain_matrix)
    out["hue"], out["sat"], out["val"] = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out["l"], out["a"], out["b"] = lab[..., 0], lab[..., 1], lab[..., 2]
    out["hem"], out["eos"], out["dab"] = hed[..., 0], hed[..., 1], hed[..., 2]
    return out, corrected


def _background_corrected(channel):
    from skimage.filters import gaussian
    sigma = max(8.0, min(channel.shape) / 10.0)
    background = gaussian(channel.astype(np.float64), sigma=sigma, preserve_range=True)
    return channel.astype(np.float64) - background


def _nucleus_mask(hem, rgb, cell):
    """Nucleus inside a drawn cell. Empty when the image shows no separate nucleus."""
    empty = np.zeros(cell.shape, bool)
    if not cell.any():
        return empty
    candidate = empty
    if hem is not None:
        candidate = _threshold_inside(hem, cell)
    fraction = float(candidate.sum()) / max(float(cell.sum()), 1.0)
    if rgb is not None and (fraction < 0.02 or fraction > 0.85):
        blue = cell & (rgb[..., 2] > rgb[..., 0] + 10) & (rgb[..., 2] > rgb[..., 1])
        if blue.any():
            candidate = blue
            fraction = float(candidate.sum()) / max(float(cell.sum()), 1.0)
    if fraction < 0.02 or fraction > 0.85:
        return empty
    minimum = max(8, int(0.01 * cell.sum()))
    candidate = _drop_holes(candidate, max(16, minimum))
    candidate = _drop_small(candidate, minimum)
    return candidate & cell


def _threshold_inside(channel, mask):
    from skimage.filters import threshold_otsu
    values = channel[mask]
    if values.size < 20 or float(np.ptp(values)) <= 1e-6:
        return np.zeros(mask.shape, bool)
    try:
        level = threshold_otsu(values)
    except ValueError:
        return np.zeros(mask.shape, bool)
    return mask & (channel >= level)


def _shape_dict(mask):
    from skimage.measure import find_contours, regionprops
    blank = _empty_shape()
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        blank["centroid_row"] = 0.0
        blank["centroid_col"] = 0.0
        return blank, np.zeros((0, 2))
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    crop = mask[r0:r1, c0:c1].astype(np.uint8)
    prop = regionprops(crop)[0]
    area = float(prop.area)
    perimeter = float(prop.perimeter)
    major = _axis_length(prop, "axis_major_length", "major_axis_length")
    minor = _axis_length(prop, "axis_minor_length", "minor_axis_length")
    contours = find_contours(crop.astype(float), 0.5)
    local = max(contours, key=len) if contours else np.zeros((0, 2))
    defects, depth = _defects(local)
    shape = {
        "area": area,
        "perimeter": perimeter,
        "equivalent_diameter": float(prop.equivalent_diameter_area),
        "major_axis": major,
        "minor_axis": minor,
        "aspect_ratio": major / minor if minor > 0 else 0.0,
        "eccentricity": float(prop.eccentricity),
        "orientation_deg": float(np.degrees(prop.orientation)),
        "solidity": float(prop.solidity),
        "extent": float(prop.extent),
        "circularity": 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0,
        "convex_area": float(prop.area_convex),
        "feret_max": float(prop.feret_diameter_max),
        "feret_min": _feret_min(local),
        "convexity_defects": defects,
        "convexity_depth": depth,
        "centroid_row": float(prop.centroid[0]),
        "centroid_col": float(prop.centroid[1]),
    }
    for index, value in enumerate(_hu(crop), start=1):
        shape[f"hu_{index}"] = value
    for (n, m), value in _zernike(crop).items():
        shape[f"zernike_n{n}_m{m}"] = value
    for index, value in enumerate(_efd(local), start=1):
        shape[f"efd_{index}"] = value
    full = local + np.array([r0, c0]) if len(local) else local
    return shape, full


def _empty_shape():
    shape = {name: "" for name in SHAPE_NAMES + _SHAPE_EXTRA}
    shape["centroid_row"] = 0.0
    shape["centroid_col"] = 0.0
    return shape


def _hu(mask):
    from skimage.measure import moments, moments_hu
    if int(mask.sum()) == 0:
        return [0.0] * 7
    values = moments_hu(moments(mask.astype(np.float64), order=3))
    out = []
    for value in values:
        magnitude = abs(float(value))
        out.append(0.0 if magnitude < 1e-30 else float(-np.sign(value) * np.log10(magnitude)))
    return out


def _zernike(mask):
    result = {(n, m): 0.0 for n, m in ZERNIKE_ORDERS}
    ys, xs = np.nonzero(mask)
    if len(ys) < 2:
        return result
    dy = ys - ys.mean()
    dx = xs - xs.mean()
    radius = float(np.sqrt(dy * dy + dx * dx).max())
    if radius < 1:
        radius = 1.0
    rho = np.sqrt(dy * dy + dx * dx) / radius
    theta = np.arctan2(dy, dx)
    for n, m in ZERNIKE_ORDERS:
        radial = _radial(n, m, rho)
        moment = (n + 1) * np.mean(radial * np.exp(-1j * m * theta))
        result[(n, m)] = float(abs(moment))
    return result


def _radial(n, m, rho):
    value = np.zeros(rho.shape, np.float64)
    for s in range((n - m) // 2 + 1):
        numerator = (-1) ** s * factorial(n - s)
        denominator = factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)
        value += (numerator / denominator) * rho ** (n - 2 * s)
    return value


def _efd(contour, order=10):
    if len(contour) < 4:
        return [0.0] * order
    points = np.asarray(contour, np.float64)
    if np.hypot(*(points[0] - points[-1])) > 1e-3:
        points = np.vstack([points, points[0]])
    delta = np.diff(points, axis=0)
    step = np.hypot(delta[:, 0], delta[:, 1])
    step[step == 0] = 1e-6
    clock = np.concatenate([[0.0], np.cumsum(step)])
    total = float(clock[-1])
    if total <= 0:
        return [0.0] * order
    magnitudes = []
    for n in range(1, order + 1):
        omega = 2 * np.pi * n / total
        scale = total / (2 * n * n * np.pi * np.pi)
        cos_delta = np.cos(omega * clock[1:]) - np.cos(omega * clock[:-1])
        sin_delta = np.sin(omega * clock[1:]) - np.sin(omega * clock[:-1])
        dx = delta[:, 1] / step
        dy = delta[:, 0] / step
        a = scale * np.sum(dx * cos_delta)
        b = scale * np.sum(dx * sin_delta)
        c = scale * np.sum(dy * cos_delta)
        d = scale * np.sum(dy * sin_delta)
        magnitudes.append(float(np.hypot(np.hypot(a, b), np.hypot(c, d))))
    first = magnitudes[0] if magnitudes[0] > 1e-8 else 1.0
    return [value / first for value in magnitudes]


def _feret_min(contour):
    if len(contour) < 2:
        return 0.0
    points = np.asarray(contour, np.float64)
    best = np.inf
    for angle in np.linspace(0, np.pi, 36, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projected = points @ direction
        best = min(best, float(projected.max() - projected.min()))
    return float(best)


def _defects(contour):
    from scipy.spatial import ConvexHull
    if len(contour) < 4:
        return 0, 0.0
    points = np.asarray(contour, np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        return 0, 0.0
    vertices = list(hull.vertices)
    count = 0
    depth = 0.0
    total = len(points)
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if start <= end:
            between = range(start + 1, end)
        else:
            between = list(range(start + 1, total)) + list(range(0, end))
        edge = points[end] - points[start]
        length = float(np.hypot(*edge))
        if length < 1 or not len(list(between) if not isinstance(between, list) else between):
            continue
        span = between if isinstance(between, list) else list(between)
        far = 0.0
        for point_index in span:
            cross = abs(edge[0] * (points[point_index, 1] - points[start, 1])
                        - edge[1] * (points[point_index, 0] - points[start, 0]))
            far = max(far, cross / length)
        if far > 1.5:
            count += 1
            depth = max(depth, far)
    return count, float(depth)


def _fill_intensity(row, channels, regions):
    from scipy.stats import kurtosis, skew
    summary = {}
    for region, mask in regions.items():
        if mask is None or not np.any(mask):
            continue
        for name, channel in channels.items():
            if channel is None:
                continue
            values = np.asarray(channel[mask], np.float64)
            if values.size == 0:
                continue
            spread = float(values.std())
            percentiles = np.percentile(values, [5, 25, 75, 95])
            stats = {
                "mean": float(values.mean()), "std": spread,
                "min": float(values.min()), "max": float(values.max()),
                "median": float(np.median(values)),
                "p5": float(percentiles[0]), "p25": float(percentiles[1]),
                "p75": float(percentiles[2]), "p95": float(percentiles[3]),
                "skew": 0.0 if spread < 1e-8 or values.size < 3 else float(skew(values)),
                "kurtosis": 0.0 if spread < 1e-8 or values.size < 4 else float(kurtosis(values)),
                "integrated": float(values.sum()),
            }
            if region == "cell":
                summary[name] = stats
            for stat, value in stats.items():
                row[f"{region}_{name}_{stat}"] = value
    gray = summary.get("gray", {})
    row["intensity_mean"] = gray.get("mean", "")
    row["intensity_min"] = gray.get("min", "")
    row["intensity_max"] = gray.get("max", "")
    row["intensity_std"] = gray.get("std", "")
    row["intensity_median"] = gray.get("median", "")
    row["integrated_intensity"] = gray.get("integrated", "")
    red, green, blue = summary.get("red", {}), summary.get("green", {}), summary.get("blue", {})
    row["mean_red"], row["std_red"] = red.get("mean", ""), red.get("std", "")
    row["mean_green"], row["std_green"] = green.get("mean", ""), green.get("std", "")
    row["mean_blue"], row["std_blue"] = blue.get("mean", ""), blue.get("std", "")


def _fill_histograms(row, channels, cell):
    for name in ("dab", "hem", "l"):
        channel = channels.get(name)
        if channel is None or not cell.any():
            continue
        counts, _ = np.histogram(channel[cell], bins=16)
        total = float(counts.sum()) or 1.0
        for index, count in enumerate(counts):
            row[f"{name}_hist_{index:02d}"] = float(count) / total


def _positive_fraction(channel, cell):
    from skimage.filters import threshold_otsu
    if channel is None or not cell.any():
        return ""
    values = channel[cell]
    if float(np.ptp(values)) <= 1e-8:
        return 0.0
    try:
        level = threshold_otsu(values)
    except ValueError:
        return 0.0
    return float(np.mean(values >= level))


def _fill_boundary(row, gray, cell):
    inner = cell & ~_eroded(cell, 3)
    outer = _dilated(cell, 3) & ~cell
    if not inner.any():
        inner = cell
    inside = float(gray[inner].mean()) if inner.any() else ""
    outside = float(gray[outer].mean()) if outer.any() else ""
    row["boundary_inside_mean"] = inside
    row["boundary_outside_mean"] = outside
    if inside != "" and outside != "":
        row["boundary_contrast"] = float(inside) - float(outside)


def _fill_puncta(row, gray, cell):
    from skimage.measure import label
    values = gray[cell]
    if values.size == 0:
        return
    center = float(np.median(values))
    spread = float(values.std()) or 1.0
    bright = _drop_small(cell & (gray > center + spread), 3)
    dark = _drop_small(cell & (gray < center - spread), 3)
    row["bright_puncta_count"] = int(label(bright).max())
    row["bright_puncta_area"] = int(bright.sum())
    row["dark_puncta_count"] = int(label(dark).max())
    row["dark_puncta_area"] = int(dark.sum())


def _fill_texture(row, name, channel, cell):
    prepared = _texture_crop(channel, cell)
    if prepared is None:
        return
    filled, mask = prepared
    _fill_glcm(row, name, filled)
    _fill_lbp(row, name, filled, mask)
    _fill_gabor(row, name, filled, mask)


def _texture_crop(channel, mask):
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    crop = channel[r0:r1, c0:c1].astype(np.float64)
    part = mask[r0:r1, c0:c1]
    filled = crop.copy()
    filled[~part] = float(np.median(crop[part]))
    if min(filled.shape) < 16:
        pad_r = 16 - filled.shape[0]
        pad_c = 16 - filled.shape[1]
        filled = np.pad(filled, ((0, max(pad_r, 0)), (0, max(pad_c, 0))), mode="edge")
        part = np.pad(part, ((0, max(pad_r, 0)), (0, max(pad_c, 0))))
    return filled, part


def _fill_glcm(row, name, filled):
    from skimage.feature import graycomatrix, graycoprops
    levels = 16
    quantised = _quantize(filled, levels)
    matrix = graycomatrix(
        quantised, distances=(1, 2, 3),
        angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels=levels, symmetric=True, normed=True,
    )
    for index, dist in enumerate((1, 2, 3)):
        slab = matrix[:, :, index:index + 1, :]
        for prop in _GLCM_PROPS:
            row[f"glcm_{name}_d{dist}_{prop}"] = float(graycoprops(slab, prop).mean())
        combined = matrix[:, :, index, :].sum(axis=2)
        total = float(combined.sum()) or 1.0
        haralick = _haralick(combined / total)
        if dist == 1:
            for number, value in enumerate(haralick, start=1):
                row[f"haralick_{name}_{number:02d}"] = value


def _haralick(probabilities):
    level = probabilities.shape[0]
    index_i, index_j = np.mgrid[0:level, 0:level]
    eps = 1e-12
    asm = float(np.sum(probabilities ** 2))
    contrast = float(np.sum((index_i - index_j) ** 2 * probabilities))
    mean_i = float(np.sum(index_i * probabilities))
    mean_j = float(np.sum(index_j * probabilities))
    var_i = float(np.sum((index_i - mean_i) ** 2 * probabilities))
    var_j = float(np.sum((index_j - mean_j) ** 2 * probabilities))
    if var_i > eps and var_j > eps:
        correlation = float(np.sum((index_i - mean_i) * (index_j - mean_j) * probabilities) / np.sqrt(var_i * var_j))
    else:
        correlation = 0.0
    inverse = float(np.sum(probabilities / (1 + (index_i - index_j) ** 2)))
    summed = np.zeros(2 * level - 1)
    differ = np.zeros(level)
    for i in range(level):
        for j in range(level):
            summed[i + j] += probabilities[i, j]
            differ[abs(i - j)] += probabilities[i, j]
    slots = np.arange(2 * level - 1)
    sum_average = float(np.sum(slots * summed))
    sum_variance = float(np.sum((slots - sum_average) ** 2 * summed))
    sum_entropy = float(-np.sum(summed * np.log2(summed + eps)))
    diff_slots = np.arange(level)
    diff_mean = float(np.sum(diff_slots * differ))
    diff_variance = float(np.sum((diff_slots - diff_mean) ** 2 * differ))
    diff_entropy = float(-np.sum(differ * np.log2(differ + eps)))
    entropy = float(-np.sum(probabilities * np.log2(probabilities + eps)))
    marginal_i = probabilities.sum(axis=1)
    marginal_j = probabilities.sum(axis=0)
    entropy_i = float(-np.sum(marginal_i * np.log2(marginal_i + eps)))
    entropy_j = float(-np.sum(marginal_j * np.log2(marginal_j + eps)))
    joint = marginal_i[:, None] * marginal_j[None, :]
    hxy1 = float(-np.sum(probabilities * np.log2(joint + eps)))
    hxy2 = float(-np.sum(joint * np.log2(joint + eps)))
    imc1 = (entropy - hxy1) / max(entropy_i, entropy_j, eps)
    imc2 = float(np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - entropy)))))
    return [asm, contrast, correlation, var_i, inverse, sum_average, sum_variance,
            sum_entropy, entropy, diff_variance, diff_entropy, imc1, imc2]


def _fill_lbp(row, name, filled, mask):
    from skimage.feature import local_binary_pattern
    codes = local_binary_pattern(_quantize(filled, 256), P=8, R=1, method="uniform")
    chosen = codes[mask]
    if chosen.size == 0:
        return
    counts, _ = np.histogram(chosen, bins=np.arange(11) - 0.5)
    total = float(counts.sum()) or 1.0
    for index, count in enumerate(counts):
        row[f"lbp_{name}_{index:02d}"] = float(count) / total


def _fill_gabor(row, name, filled, mask):
    from skimage.filters import gabor
    from skimage.transform import resize
    image, part = filled, mask
    if max(image.shape) > 64:
        scale = 64 / max(image.shape)
        size = (max(8, int(round(image.shape[0] * scale))), max(8, int(round(image.shape[1] * scale))))
        image = resize(image, size, preserve_range=True, anti_aliasing=True)
        part = resize(mask.astype(float), size, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    if not part.any():
        part = np.ones(image.shape, bool)
    for scale_index, frequency in enumerate((0.1, 0.2), start=1):
        for angle in (0, 45, 90, 135):
            real, imag = gabor(image, frequency=frequency, theta=np.deg2rad(angle))
            energy = np.sqrt(real ** 2 + imag ** 2)
            row[f"gabor_{name}_a{angle}_s{scale_index}"] = float(energy[part].mean())


def _quantize(image, levels):
    low, high = np.percentile(image, [1, 99])
    if not np.isfinite(low) or high <= low:
        return np.zeros(image.shape, np.uint8)
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return np.clip(np.round(scaled * (levels - 1)), 0, levels - 1).astype(np.uint8)


def _eroded(mask, radius):
    from skimage.morphology import disk
    return _morph(mask, lambda crop: _morphology(crop, disk(radius), shrink=True), radius)


def _dilated(mask, radius):
    from skimage.morphology import disk
    return _morph(mask, lambda crop: _morphology(crop, disk(radius), shrink=False), radius)


def _morphology(crop, footprint, shrink):
    image = np.asarray(crop, bool)
    if shrink:
        from skimage.morphology import erosion
        try:
            out = erosion(image, footprint, mode="constant", cval=0)
        except TypeError:
            from skimage.morphology import binary_erosion
            out = binary_erosion(image, footprint)
    else:
        from skimage.morphology import dilation
        try:
            out = dilation(image, footprint, mode="constant", cval=0)
        except TypeError:
            from skimage.morphology import binary_dilation
            out = binary_dilation(image, footprint)
    return np.asarray(out, bool)


def _drop_small(mask, minimum):
    from skimage.morphology import remove_small_objects
    if minimum <= 0:
        return np.asarray(mask, bool)
    try:
        return remove_small_objects(np.asarray(mask, bool), max_size=minimum - 1)
    except TypeError:
        return remove_small_objects(np.asarray(mask, bool), min_size=minimum)


def _drop_holes(mask, minimum):
    from skimage.morphology import remove_small_holes
    if minimum <= 0:
        return np.asarray(mask, bool)
    try:
        return remove_small_holes(np.asarray(mask, bool), max_size=minimum - 1)
    except TypeError:
        return remove_small_holes(np.asarray(mask, bool), area_threshold=minimum)


def _morph(mask, operation, radius):
    ys, xs = np.nonzero(mask)
    out = np.zeros(mask.shape, bool)
    if len(ys) == 0:
        return out
    pad = radius + 1
    r0, c0 = max(0, int(ys.min()) - pad), max(0, int(xs.min()) - pad)
    r1, c1 = min(mask.shape[0], int(ys.max()) + 1 + pad), min(mask.shape[1], int(xs.max()) + 1 + pad)
    out[r0:r1, c0:c1] = operation(mask[r0:r1, c0:c1])
    return out


def _bounds(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _object_count(mask):
    from skimage.measure import label
    return int(label(mask).max())


def _rle(mask):
    flat = np.asarray(mask, np.uint8).ravel()
    padded = np.concatenate([[0], flat, [0]])
    change = np.diff(padded.astype(np.int16))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    pairs = []
    for start, end in zip(starts, ends):
        pairs.extend((int(start), int(end - start)))
    return pairs


def _contour_xy(contour):
    if len(contour) == 0:
        return []
    points = np.asarray(contour, np.float64)
    if len(points) > 800:
        points = points[::max(1, len(points) // 800)]
    return [[round(float(col), 2), round(float(row), 2)] for row, col in points]


def _letterbox(patch, size):
    from skimage.transform import resize
    patch = np.asarray(patch)
    if patch.ndim == 2:
        patch = np.stack([patch, patch, patch], axis=-1)
    height, width = patch.shape[:2]
    scale = size / max(height, width, 1)
    resized_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    resized = resize(patch, resized_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    canvas = np.zeros((size, size, 3), np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _display_rgb(image):
    image = np.asarray(image)
    source = image[..., :3] if image.ndim == 3 and image.shape[-1] in (3, 4) else image
    if source.dtype == np.uint8:
        shown = source
    else:
        low, high = np.percentile(source, [1, 99])
        if high <= low:
            high = low + 1
        shown = np.clip((source.astype(np.float64) - low) / (high - low), 0, 1)
        shown = (shown * 255).astype(np.uint8)
    if shown.ndim == 2:
        shown = np.stack([shown, shown, shown], axis=-1)
    return np.ascontiguousarray(shown)


def _original_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].astype(np.float64)
    return None


def _split_image(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(np.float64)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return gray, rgb
    if image.ndim != 2:
        raise ValueError(f"Features need a 2-D or colour image, not shape {image.shape}.")
    return image.astype(np.float64), None


def _axis_length(prop, current, previous):
    try:
        return float(getattr(prop, current))
    except AttributeError:
        return float(getattr(prop, previous))


def _write_png(path, array):
    import imageio.v2 as imageio
    picture = np.asarray(array)
    atomic_write(path, lambda target: imageio.imwrite(target, picture))


def _read_png(path):
    import imageio.v2 as imageio
    return np.asarray(imageio.imread(path))


def _mask_png(mask):
    return (np.asarray(mask, bool).astype(np.uint8) * 255)


def _write_qc(path, rgb, labels, nuclei, classes):
    canvas = np.array(rgb, copy=True)
    for lab in [int(i) for i in np.unique(labels) if int(i) > 0]:
        color = _class_color(classes.get(lab, ""))
        _paint_contour(canvas, labels == lab, color, width=2)
        if np.any(nuclei == lab):
            _paint_contour(canvas, nuclei == lab, (255, 255, 255), width=1)
        ys, xs = np.nonzero(labels == lab)
        _draw_label(canvas, int(xs.mean()), int(ys.mean()), str(lab), color)
    _write_png(path, canvas)


def _paint_contour(image, mask, color, width=1):
    if mask is None or not np.any(mask):
        return
    edge = mask & ~_eroded(mask, 1)
    if width > 1:
        edge = _dilated(edge, width - 1)
    image[edge] = color


def _draw_label(image, x, y, text, color):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    picture = Image.fromarray(image)
    ImageDraw.Draw(picture).text((x + 2, y + 2), text, fill=tuple(int(v) for v in color))
    image[...] = np.asarray(picture)


def _class_color(name):
    key = (name or "").strip().lower()
    named = {
        "infected": (220, 70, 70), "positive": (220, 70, 70), "dead": (150, 70, 170),
        "uninfected": (70, 180, 110), "negative": (70, 180, 110), "healthy": (70, 180, 110),
    }
    if key in named:
        return np.array(named[key], np.uint8)
    return np.array(_CLASS_COLORS[zlib.crc32(key.encode()) % len(_CLASS_COLORS)], np.uint8)


def _read_rows(folder):
    folder = Path(folder)
    csv_path = folder if folder.suffix.lower() == ".csv" else folder / "features.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_for(folder, label):
    for row in _read_rows(folder):
        if int(float(row["label"])) == int(label):
            return row
    raise KeyError(f"No cell {label} in {folder}")


_PHYSICAL = (
    ("area", "area_um2", 2),
    ("perimeter", "perimeter_um", 1),
    ("equivalent_diameter", "equivalent_diameter_um", 1),
    ("major_axis", "major_axis_um", 1),
    ("minor_axis", "minor_axis_um", 1),
    ("feret_max", "feret_max_um", 1),
    ("feret_min", "feret_min_um", 1),
)


def _apply_pixel_size(row: dict, pixel_size_um) -> None:
    """Convert pixel measurements when the user typed a micrometre scale."""
    try:
        scale = float(str(pixel_size_um).replace(",", "."))
    except (TypeError, ValueError):
        return
    if not np.isfinite(scale) or scale <= 0:
        return
    row["pixel_size_um"] = scale
    for source, dest, power in _PHYSICAL:
        value = row.get(source, "")
        if value == "" or value is None:
            continue
        row[dest] = float(value) * (scale ** power)


def _safe_stem(image_name):
    stem = Path(image_name).stem if image_name else "image"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return cleaned or "image"


def _write_readme(path):
    lines = [
        "Ani cell features",
        "",
        "These rows describe the cells you outlined. The outlines are not replaced by an automatic segmenter.",
        "On a colour image, the nucleus is estimated inside each cell from the haematoxylin channel of an HED colour deconvolution (skimage rgb2hed, Ruifrok matrix).",
        "Pass stain_matrix to export_training_set if the stain is not haematoxylin/eosin/DAB.",
        "Cytoplasm is the outlined cell minus that nucleus. A grey image leaves nucleus and colour fields empty.",
        "Patches and masks use the original pixels, with 10 px of padding. Intensity, colour and texture are measured on those same pixels.",
        "gray_corrected_mean subtracts a large Gaussian first, so you can compare cells when the lighting is uneven.",
        "touches_border is 1 when the cell is cut by the image edge. neighbour_count counts other cells whose centroid is within two equivalent diameters.",
        "Hu moments are signed log10 values. efd_1 to efd_10 are scale-normalised elliptic Fourier harmonic magnitudes.",
        "Zernike columns are magnitudes for orders 0 to 4. Haralick columns are the 13 co-occurrence features at distance 1, averaged over 4 angles.",
        "GLCM columns are averaged over 4 angles at distances 1, 2 and 3. LBP is the uniform 8-neighbour histogram. Gabor uses 4 orientations and 2 frequencies.",
        "Histograms are 16 bins, scaled so each histogram sums to 1. stain_positive_fraction is the share of the cell above an Otsu threshold on the DAB channel.",
        "boundary_contrast is the mean grey value in a 3 px ring inside the outline minus the ring just outside it.",
        "rle is a JSON list of [start, length] pairs for the tight bounding box, row by row. Decode it with mask_from_rle.",
        "contour_json is a JSON list of [x, y] points in full-image pixel coordinates.",
        "When pixel_size_um is set, area_um2 and the *_um lengths are the same measurements in micrometres.",
        "reconstruct_cell(folder, label) draws the saved cell and nucleus contours on the patch.",
        "paste_cells(folder) puts every cell back on a blank image at its original position, coloured by class.",
        "",
        "Files: features.csv, patches/, masks/, cnn/, qc/.",
        "",
        "Columns",
    ]
    for name in FEATURE_COLUMNS:
        lines.append(f"{name}: {_describe(name)}")
    text = "\n".join(lines) + "\n"
    atomic_write(path, lambda target: target.write_text(text, encoding="utf-8"))


def _describe(name):
    notes = {
        "image": "File name of the source image.",
        "well": "Well id typed in Ani, if any.",
        "slide": "Slide id typed in Ani, if any.",
        "magnification": "Magnification typed in Ani, if any.",
        "pixel_size_um": "Micrometres per pixel typed in Ani. Empty when the image is not calibrated.",
        "area_um2": "Cell area in square micrometres. Empty when pixel size is blank.",
        "perimeter_um": "Cell perimeter in micrometres.",
        "equivalent_diameter_um": "Diameter of a circle with the same area, in micrometres.",
        "major_axis_um": "Major axis in micrometres.",
        "minor_axis_um": "Minor axis in micrometres.",
        "feret_max_um": "Longest caliper diameter in micrometres.",
        "feret_min_um": "Shortest caliper diameter in micrometres.",
        "label": "Cell id in the outline mask. 0 is background.",
        "class": "Class you assigned, such as infected or uninfected. This is the training label.",
        "image_height": "Height of the source image in pixels.",
        "image_width": "Width of the source image in pixels.",
        "touches_border": "1 if the cell touches the image edge.",
        "neighbour_count": "Other cells within two equivalent diameters.",
        "bbox_row": "Top row of the tight cell box.",
        "bbox_col": "Left column of the tight cell box.",
        "bbox_height": "Tight box height.",
        "bbox_width": "Tight box width.",
        "centroid_row": "Cell centroid row.",
        "centroid_col": "Cell centroid column.",
        "pad": "Padding, in pixels, added around the box in the saved crops.",
        "crop_row": "Top row of the padded crop in the full image.",
        "crop_col": "Left column of the padded crop in the full image.",
        "crop_height": "Padded crop height.",
        "crop_width": "Padded crop width.",
        "contour_json": "Outline as JSON [x, y] points in full-image coordinates.",
        "rle": "Run-length backup of the tight cell mask.",
        "rle_height": "Height of the RLE mask.",
        "rle_width": "Width of the RLE mask.",
        "patch": "RGB crop, relative to this folder.",
        "mask_cell": "Cell mask for that crop.",
        "mask_nucleus": "Nucleus mask for that crop.",
        "mask_cytoplasm": "Cytoplasm mask for that crop.",
        "cnn_patch": "128x128 training image, aspect preserved, black padding.",
        "nucleus_count": "How many separate nuclei were found inside the cell.",
        "nucleus_cell_area_ratio": "Nucleus area divided by cell area.",
        "nucleus_offset_row": "Nucleus centroid row minus cell centroid row.",
        "nucleus_offset_col": "Nucleus centroid column minus cell centroid column.",
        "gray_corrected_mean": "Mean brightness after large-Gaussian background removal.",
        "stain_positive_fraction": "Fraction of the cell above the DAB Otsu threshold.",
        "boundary_inside_mean": "Mean brightness in the 3 px ring inside the outline.",
        "boundary_outside_mean": "Mean brightness in the 3 px ring outside the outline.",
        "boundary_contrast": "Inside ring minus outside ring.",
        "bright_puncta_count": "Number of bright spots inside the cell.",
        "bright_puncta_area": "Pixel area of those bright spots.",
        "dark_puncta_count": "Number of dark spots or vacuoles inside the cell.",
        "dark_puncta_area": "Pixel area of those dark spots.",
        "feret_diameter": "Longest caliper diameter of the cell. Same as feret_max.",
        "intensity_mean": "Mean grey value of the whole cell.",
        "intensity_min": "Minimum grey value of the whole cell.",
        "intensity_max": "Maximum grey value of the whole cell.",
        "intensity_std": "Standard deviation of grey values in the cell.",
        "intensity_median": "Median grey value of the cell.",
        "integrated_intensity": "Sum of grey values in the cell.",
        "mean_red": "Mean red in the cell. Empty for a grey image.",
        "mean_green": "Mean green in the cell.",
        "mean_blue": "Mean blue in the cell.",
        "std_red": "Standard deviation of red in the cell.",
        "std_green": "Standard deviation of green in the cell.",
        "std_blue": "Standard deviation of blue in the cell.",
    }
    if name in notes:
        return notes[name]
    if name.startswith("nuc_"):
        return "Same measurement as " + name[4:] + ", for the largest nucleus. Empty when no nucleus was found."
    if name.startswith("hu_"):
        return "Signed log10 Hu moment " + name.split("_")[1] + " of the cell."
    if name.startswith("zernike_"):
        return "Zernike moment magnitude " + name + " of the cell."
    if name.startswith("efd_"):
        return "Scale-normalised elliptic Fourier magnitude " + name.split("_")[1] + " of the cell."
    if name in SHAPE_NAMES:
        return "Cell shape: " + name.replace("_", " ") + "."
    if "_hist_" in name:
        channel, index = name.split("_hist_")
        return f"Normalised 16-bin histogram bin {index} of the {channel} channel inside the cell."
    if name.startswith("glcm_"):
        return "GLCM " + name + ", averaged over 4 angles."
    if name.startswith("haralick_"):
        return "Haralick feature " + name + " from the distance-1 co-occurrence matrix."
    if name.startswith("lbp_"):
        return "Uniform local-binary-pattern histogram bin for " + name + "."
    if name.startswith("gabor_"):
        return "Mean Gabor energy " + name + " inside the cell."
    for region in REGIONS:
        for channel in CHANNELS:
            prefix = f"{region}_{channel}_"
            if name.startswith(prefix):
                stat = name[len(prefix):]
                return f"{stat} of {channel} in the {region}."
    return name


def _cell(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        return round(value, 6)
    return value
