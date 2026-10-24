"""Lane densitometry with Fiji-style optical density and core band integration.

Principles from ImageJ/Fiji gel analysis and classical gel tools
(Gel-Blot-Analyzer valley-to-valley, GelGenie band focus) without bundling
those stacks:

* Rolling-ball / morphological background, then absorbance
* Profile from high-percentile band core (not washed-out means)
* Valley-to-valley notch baseline for peak height and area
* Auto profile direction from ROI aspect ratio
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, grey_closing, grey_opening
from scipy.signal import find_peaks

from .segment import to_gray

_OD_EPS = 1e-4
# High percentile = darkest ink in the cross-section (doublets / white margins out).
_CORE_PERCENTILE = 90.0


@dataclass(frozen=True)
class Peak:
    """One band on the densitogram."""

    number: int
    position: float
    intensity: float
    level: float
    area: float
    spread: float
    left: int
    right: int
    sample: int


@dataclass(frozen=True)
class DensityResult:
    """A lane profile with its notch baseline and the peaks that sit on it."""

    profile: np.ndarray
    baseline: np.ndarray
    peaks: list[Peak]
    origin: float
    along: str
    roi: tuple[int, int, int, int]
    dark_bands: bool
    notch_percent: float


def luminance(image: np.ndarray) -> np.ndarray:
    """Brightness of a 2-D or colour image, as float."""
    return np.asarray(to_gray(image), np.float64)


def clamp_roi(roi, shape) -> tuple[int, int, int, int]:
    """(row0, col0, row1, col1) inside the image. None means the whole image."""
    height, width = int(shape[0]), int(shape[1])
    if roi is None:
        return (0, 0, height, width)
    r0, c0, r1, c1 = (int(round(float(v))) for v in roi)
    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))
    return (max(0, r0), max(0, c0), min(height, r1), min(width, c1))


def resolve_along(along: str | None, roi: tuple[int, int, int, int]) -> str:
    """Pick profile direction. Wide boxes (blot rows) read across; tall lanes read down."""
    if along in (None, "", "auto"):
        r0, c0, r1, c1 = roi
        return "horizontal" if (c1 - c0) >= 1.2 * max(r1 - r0, 1) else "vertical"
    return "horizontal" if along == "horizontal" else "vertical"


def optical_density_map(
    gray: np.ndarray,
    dark_bands: bool = True,
    ball: int | None = None,
) -> np.ndarray:
    """Fiji-style densitometry after rolling-ball background estimation.

    Morphological closing (light background) estimates the gel plate. Density is:

    * Linear ``I0 - I`` so saturated blacks keep ranking by how much ink is there
    * Soft ``-log10(I/I0)`` so faint mid-tone bands still rise above noise

    Higher values = darker / denser stain.
    """
    g = np.asarray(gray, np.float64)
    if g.ndim != 2 or g.size == 0:
        return np.zeros_like(g, dtype=np.float64)
    height, width = g.shape
    radius = int(ball) if ball is not None else int(np.clip(min(height, width) / 5.0, 6, 48))
    size = max(3, 2 * (radius // 2) + 1)
    soft = max(0.8, radius / 10.0)
    if dark_bands:
        background = grey_closing(g, size=(size, size))
        background = gaussian_filter(background, sigma=soft)
        border = _border_background(g)
        background = np.maximum(background, border * 0.95)
        linear = np.clip(background - g, 0.0, None)
        ratio = np.clip(g / np.clip(background, 1.0, None), _OD_EPS, 1.0)
        od = -np.log10(ratio)
        return (linear + od * (border / 4.0)).astype(np.float64, copy=False)
    background = grey_opening(g, size=(size, size))
    background = gaussian_filter(background, sigma=soft)
    border = float(np.percentile(g, 10)) if g.size else 1.0
    background = np.minimum(background, max(border, 1.0))
    linear = np.clip(g - background, 0.0, None)
    ratio = np.clip(np.clip(background, 1.0, None) / np.clip(g, 1.0, None), _OD_EPS, 1.0)
    od = -np.log10(ratio)
    return (linear + od * (max(border, 1.0) / 4.0)).astype(np.float64, copy=False)


def _border_background(strip: np.ndarray) -> float:
    """Gel-plate level from the ROI rim."""
    if strip.size < 4:
        return float(np.max(strip)) if strip.size else 255.0
    h, w = strip.shape
    rim = max(1, min(h, w) // 10)
    parts = [strip[:rim, :], strip[-rim:, :], strip[:, :rim], strip[:, -rim:]]
    samples = np.concatenate([p.ravel() for p in parts])
    return float(np.percentile(samples, 90))


def _stain(mask, shape, roi):
    """Stained pixels inside the selection, or None when the whole band counts."""
    if mask is None:
        return None
    mask = np.asarray(mask, bool)
    if mask.shape[:2] != tuple(shape[:2]):
        return None
    r0, c0, r1, c1 = roi
    return mask[r0:r1, c0:c1]


def _core_profile(density: np.ndarray, stain, vertical: bool, percentile: float = _CORE_PERCENTILE):
    """High-percentile density along the short axis — tracks band darkness.

    Using a percentile (not a sum) matches visual ink darkness left→right on
    western blots. Peak *area* still integrates quantity between notches.
    """
    length = density.shape[0] if vertical else density.shape[1]
    profile = np.full(length, np.nan, np.float64)
    pct = float(np.clip(percentile, 50.0, 100.0))
    for index in range(length):
        if vertical:
            line = density[index, :]
            keep = None if stain is None else stain[index, :]
        else:
            line = density[:, index]
            keep = None if stain is None else stain[:, index]
        if keep is not None:
            line = line[np.asarray(keep, bool)]
        if line.size == 0:
            continue
        profile[index] = float(np.percentile(line, pct))
    valid = np.isfinite(profile)
    if int(valid.sum()) < 4:
        return np.zeros(0, np.float64), 0
    start = int(np.flatnonzero(valid)[0])
    end = int(np.flatnonzero(valid)[-1]) + 1
    cropped = np.asarray(profile[start:end], np.float64)
    if np.isnan(cropped).any():
        index = np.arange(cropped.size)
        good = np.isfinite(cropped)
        cropped = np.interp(index, index[good], cropped[good])
    return cropped, start


def _masked_mean_profile(density: np.ndarray, stain, vertical: bool):
    """Full-width mean (tiny masks / regression safety)."""
    axis = 1 if vertical else 0
    if stain is None:
        return np.asarray(density.mean(axis=axis), np.float64), 0
    counts = np.sum(stain, axis=axis)
    totals = np.sum(np.where(stain, density, 0.0), axis=axis)
    profile = np.full(counts.shape, np.nan, np.float64)
    np.divide(totals, counts, out=profile, where=counts > 0)
    valid = np.isfinite(profile)
    if int(valid.sum()) < 4:
        return np.zeros(0, np.float64), 0
    start = int(np.flatnonzero(valid)[0])
    end = int(np.flatnonzero(valid)[-1]) + 1
    cropped = np.asarray(profile[start:end], np.float64)
    if np.isnan(cropped).any():
        index = np.arange(cropped.size)
        good = np.isfinite(cropped)
        cropped = np.interp(index, index[good], cropped[good])
    return cropped, start


def analyze_lane(
    image,
    roi=None,
    along: str = "auto",
    dark_bands: bool = True,
    notch_percent: float = 8.0,
    mask=None,
) -> DensityResult:
    """Profile one selection with Fiji-style OD and split peaks at troughs.

    ``along="auto"`` reads across wide blot rows and down tall lanes.
    Table intensity is absolute core density at the apex (tracks band darkness).
    """
    gray = luminance(image)
    r0, c0, r1, c1 = clamp_roi(roi, gray.shape)
    direction = resolve_along(along, (r0, c0, r1, c1))
    vertical = direction != "horizontal"
    empty = DensityResult(
        np.zeros(0, np.float64), np.zeros(0, np.float64), [], 0.0,
        direction, (r0, c0, r1, c1),
        bool(dark_bands), float(notch_percent),
    )
    if r1 - r0 < 2 or c1 - c0 < 2:
        return empty
    # Pad the short axis so thick/saturated bands are not clipped mid-band.
    pr0, pr1, pc0, pc1 = r0, r1, c0, c1
    if vertical:
        pad = max(1, (c1 - c0) // 8)
        pc0, pc1 = max(0, c0 - pad), min(gray.shape[1], c1 + pad)
    else:
        pad = max(2, (r1 - r0) // 3)
        pr0, pr1 = max(0, r0 - pad), min(gray.shape[0], r1 + pad)
    strip = gray[pr0:pr1, pc0:pc1]
    stain = _stain(mask, gray.shape, (pr0, pc0, pr1, pc1))
    if stain is not None and not np.any(stain):
        return empty
    density = optical_density_map(strip, dark_bands=dark_bands)
    if stain is not None and int(stain.sum()) < 24:
        profile, offset = _masked_mean_profile(density, stain, vertical)
    else:
        profile, offset = _core_profile(density, stain, vertical)
    origin = float((pr0 if vertical else pc0) + offset)
    if profile.size < 4:
        return empty
    peaks, baseline = _peaks_and_notches(
        profile, origin, notch_percent, density=density, vertical=vertical,
    )
    if not peaks:
        peaks, baseline = _single_band_peak(profile, density, stain, origin)
    else:
        # Re-measure intensity from the 2D band core (Fiji-style), not the 1D spike.
        peaks = [
            Peak(
                number=peak.number,
                position=peak.position,
                intensity=peak.intensity,
                level=_band_core_level(
                    density, peak.left, peak.right, vertical,
                    profile=profile, apex=peak.sample,
                ) or peak.level,
                area=peak.area,
                spread=peak.spread,
                left=peak.left,
                right=peak.right,
                sample=peak.sample,
            )
            for peak in peaks
        ]
    return DensityResult(
        profile, baseline, peaks, origin, direction,
        (r0, c0, r1, c1), bool(dark_bands), float(notch_percent),
    )


def _single_band_peak(profile, density, stain, origin: float):
    """One box around one band: report core OD even when no trough exists."""
    vals = density[stain] if stain is not None else density.ravel()
    if vals.size == 0:
        return [], np.full(profile.shape, np.nan, np.float64)
    level = float(np.percentile(vals, _CORE_PERCENTILE))
    if level <= 1e-9:
        return [], np.full(profile.shape, np.nan, np.float64)
    apex = int(np.argmax(profile))
    baseline = np.zeros(profile.shape, np.float64)
    above = np.clip(profile - baseline, 0, None)
    peak = Peak(
        number=1,
        position=float(origin + apex),
        intensity=float(above[apex]) if above[apex] > 0 else level,
        level=level,
        area=_trapz(above) if float(above.sum()) > 0 else float(vals.sum()),
        spread=float(max(profile.size - 1, 1)),
        left=0,
        right=int(profile.size - 1),
        sample=apex,
    )
    return [peak], baseline


def surface_mesh(image, roi=None, dark_bands: bool = True, max_side: int = 160, mask=None):
    """Height field of the lane. Vertices are (height, row, column) for napari."""
    gray = luminance(image)
    r0, c0, r1, c1 = clamp_roi(roi, gray.shape)
    strip = gray[r0:r1, c0:c1]
    if strip.shape[0] < 2 or strip.shape[1] < 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    stain = _stain(mask, gray.shape, (r0, c0, r1, c1))
    if stain is not None and not np.any(stain):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    density = optical_density_map(strip, dark_bands=dark_bands)
    if stain is not None:
        density = np.where(stain, density, 0.0)
    step_r = max(1, int(np.ceil(density.shape[0] / max_side)))
    step_c = max(1, int(np.ceil(density.shape[1] / max_side)))
    small = np.asarray(density[::step_r, ::step_c], np.float64)
    rows_n, cols_n = small.shape
    if rows_n < 2 or cols_n < 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    measured = small[small > 0] if stain is not None else small
    if measured.size == 0:
        measured = small
    low, high = np.percentile(measured, [2, 98])
    span = max(float(high - low), 1e-6)
    norm = np.clip((small - low) / span, 0, 1)
    rise = 0.55 * max(rows_n * step_r, cols_n * step_c, 1)
    height = norm * rise
    rows = r0 + np.arange(rows_n) * step_r
    cols = c0 + np.arange(cols_n) * step_c
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    vertices = np.column_stack([
        height.ravel(), rr.ravel().astype(np.float64), cc.ravel().astype(np.float64),
    ]).astype(np.float32)
    stained = None
    if stain is not None:
        stained = np.asarray(stain[::step_r, ::step_c], bool)[:rows_n, :cols_n]
    faces_list = []
    for i in range(rows_n - 1):
        for j in range(cols_n - 1):
            if stained is not None and not (
                stained[i, j] and stained[i + 1, j] and stained[i, j + 1] and stained[i + 1, j + 1]
            ):
                continue
            a = i * cols_n + j
            b = a + 1
            c = a + cols_n
            d = c + 1
            faces_list.append((a, c, b))
            faces_list.append((b, c, d))
    if not faces_list:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    return vertices, np.asarray(faces_list, np.int32), small.ravel().astype(np.float32)


def combine_surfaces(parts):
    """Stack one mesh per area. Faces never cross from one area into another."""
    vertices, faces, values = [], [], []
    offset = 0
    for mesh in parts:
        verts, tris, vals = mesh
        if tris.size == 0:
            continue
        vertices.append(np.asarray(verts, np.float32))
        faces.append(np.asarray(tris, np.int32) + offset)
        values.append(np.asarray(vals, np.float32))
        offset += len(verts)
    if not vertices:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(values)


def _band_core_level(
    density: np.ndarray,
    left: int,
    right: int,
    vertical: bool,
    profile: np.ndarray | None = None,
    apex: int | None = None,
) -> float:
    """Mean OD of the densest ~30% pixels near the apex (tight window, not full notch span)."""
    if density is None or density.size == 0 or right < left:
        return 0.0
    # Fixed half-width around the apex (not full notch span) so wide valleys
    # between blot bands do not dilute the rightmost saturated cores.
    if profile is not None and apex is not None and 0 <= int(apex) < len(profile):
        apex_i = int(apex)
        half = 12
        span = density.shape[0] if vertical else density.shape[1]
        lo = max(0, apex_i - half)
        hi = min(span - 1, apex_i + half)
    else:
        lo, hi = int(left), int(right)
    if hi < lo:
        lo, hi = int(left), int(right)
    if vertical:
        region = density[lo:hi + 1, :]
    else:
        region = density[:, lo:hi + 1]
    flat = np.asarray(region, np.float64).ravel()
    if flat.size == 0:
        return 0.0
    floor = float(np.percentile(flat, 70.0))
    core = flat[flat >= floor]
    if core.size == 0:
        core = flat
    return float(core.mean())


def _peaks_and_notches(
    profile: np.ndarray,
    origin: float,
    notch_percent: float,
    density: np.ndarray | None = None,
    vertical: bool = True,
):
    """Split the profile on valleys; drop doublet shoulders inside one band."""
    sigma = float(np.clip(profile.size / 90.0, 0.9, 3.5))
    smooth = gaussian_filter1d(profile, sigma)
    span = float(np.ptp(smooth))
    baseline = np.full(profile.shape, np.nan, np.float64)
    if span <= 1e-9:
        return [], baseline
    prominence = max(span * float(np.clip(notch_percent, 1.0, 80.0)) / 100.0 * 0.55, 1e-6)
    distance = max(8, int(profile.size / 14))
    apices, props = find_peaks(smooth, prominence=prominence, distance=distance)
    if apices.size < 2:
        apices, props = find_peaks(
            smooth,
            prominence=max(prominence * 0.35, 1e-6),
            distance=max(5, distance // 2),
        )
    if apices.size == 0:
        return [], baseline
    apices = _drop_doublet_shoulders(smooth, apices, props.get("prominences"))
    if apices.size == 0:
        return [], baseline
    notches = _notch_bounds(smooth, apices)
    peaks: list[Peak] = []
    for number, (apex, left, right) in enumerate(zip(apices, notches[:-1], notches[1:]), start=1):
        if right <= left:
            continue
        segment = smooth[left:right + 1]
        line = np.linspace(float(smooth[left]), float(smooth[right]), segment.size)
        above = np.clip(segment - line, 0, None)
        apex_at = int(np.clip(apex - left, 0, segment.size - 1))
        intensity = float(above[apex_at])
        if intensity <= 0:
            continue
        baseline[left:right + 1] = line
        if density is not None:
            level = _band_core_level(
                density, left, right, vertical, profile=smooth, apex=int(apex),
            )
        else:
            level = float(smooth[int(apex)])
        if level <= 1e-9:
            level = float(smooth[int(apex)])
        peaks.append(Peak(
            number=number,
            position=float(origin + int(apex)),
            intensity=intensity,
            level=level,
            area=_trapz(above),
            spread=float(right - left),
            left=int(left),
            right=int(right),
            sample=int(apex),
        ))
    return peaks, baseline


def _drop_doublet_shoulders(smooth: np.ndarray, apices: np.ndarray, prominences) -> np.ndarray:
    """Keep the stronger apex when two peaks sit inside one blot band."""
    if apices.size < 2:
        return apices
    prom = np.asarray(prominences if prominences is not None else smooth[apices], np.float64)
    keep = np.ones(apices.size, bool)
    min_gap = max(6, int(smooth.size / 12))
    for i in range(apices.size - 1):
        if not keep[i]:
            continue
        for j in range(i + 1, apices.size):
            if not keep[j]:
                continue
            if int(apices[j] - apices[i]) > min_gap:
                break
            if prom[i] >= prom[j]:
                keep[j] = False
            else:
                keep[i] = False
                break
    return apices[keep]


def _notch_bounds(smooth: np.ndarray, apices: np.ndarray) -> list[int]:
    """Valley index on each side of every peak, including the outer shoulders."""
    bounds = [_outer_notch(smooth, int(apices[0]), -1)]
    for left, right in zip(apices[:-1], apices[1:]):
        left, right = int(left), int(right)
        bounds.append(left + int(np.argmin(smooth[left:right + 1])))
    bounds.append(_outer_notch(smooth, int(apices[-1]), 1))
    return bounds


def _outer_notch(smooth: np.ndarray, apex: int, direction: int) -> int:
    """Walk away from the peak until the profile starts rising again."""
    index = apex
    last = len(smooth) - 1
    while 0 < index < last:
        nxt = index + direction
        if smooth[nxt] > smooth[index]:
            break
        index = nxt
    return index


def _trapz(height: np.ndarray) -> float:
    if height.size < 2:
        return float(height.sum()) if height.size else 0.0
    return float(np.sum((height[:-1] + height[1:]) * 0.5))


def density_table_rows(results: list[DensityResult], names: list[str] | None = None) -> list[dict]:
    """One row per peak for CSV export."""
    rows: list[dict] = []
    for number, result in enumerate(results, start=1):
        name = ""
        if names is not None and number - 1 < len(names):
            name = str(names[number - 1] or "")
        for peak in result.peaks:
            rows.append({
                "roi": number,
                "name": name,
                "peak": peak.number,
                "intensity": peak.level,
                "height": peak.intensity,
                "area": peak.area,
                "spread_px": peak.spread,
                "position": peak.position,
            })
    return rows


def write_density_csv(path, results: list[DensityResult], names: list[str] | None = None):
    """Write the densitogram table. Returns the Path that was written."""
    import csv
    from pathlib import Path

    path = Path(path)
    rows = density_table_rows(results, names)
    fields = ["roi", "name", "peak", "intensity", "height", "area", "spread_px", "position"]

    def write(file):
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fields})

    from .safety import atomic_write

    return atomic_write(path, write)
