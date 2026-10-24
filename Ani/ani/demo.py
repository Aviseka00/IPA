"""Make a small demo project with synthetic cells and a deliberately imperfect 'detector' output,
so new users can practise corrections without their own data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import io
from .project import Project


def _synthetic(seed: int, size: int = 384, n: int = 24):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    truth = np.zeros((size, size), np.uint16)
    centres = []
    while len(centres) < n:
        c = rng.uniform(25, size - 25, 2)
        if all(np.hypot(*(c - o)) > 30 for o in centres):
            centres.append(c)
    for i, (cy, cx) in enumerate(centres, 1):
        a, b, t = rng.uniform(9, 15), rng.uniform(7, 12), rng.uniform(0, np.pi)
        y, x = yy - cy, xx - cx
        inside = ((x * np.cos(t) + y * np.sin(t)) / a) ** 2 + ((-x * np.sin(t) + y * np.cos(t)) / b) ** 2 <= 1
        truth[inside & (truth == 0)] = i
    signal = ndi.gaussian_filter((truth > 0).astype(float) * rng.uniform(0.6, 1.0, n + 1)[truth], 2)
    img = 200 + 3000 * signal + rng.normal(0, 60, signal.shape) + 400 * ndi.gaussian_filter(rng.random(signal.shape), 40)
    img = np.clip(img, 0, 65535).astype(np.uint16)
    return img, truth, centres


def _imperfect(truth: np.ndarray, centres, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    pred = truth.copy()
    cents = np.array(centres)                   # pick two NEIGHBOURING cells to merge
    d = np.hypot(*(cents[:, None, :] - cents[None, :, :]).transpose(2, 0, 1)) + np.eye(len(cents)) * 1e9
    a, b = (int(i) + 1 for i in np.unravel_index(np.argmin(d), d.shape))
    ids = [i for i in rng.permutation(np.arange(1, truth.max() + 1)) if i not in (a, b)]
    pred[pred == ids[0]] = 0                    # missed cell
    pred[pred == ids[1]] = 0                    # another missed cell
    from skimage.draw import line
    rr, cc = line(*np.round(cents[a - 1]).astype(int), *np.round(cents[b - 1]).astype(int))
    bridge = np.zeros(truth.shape, bool)
    bridge[rr, cc] = True
    bridge = ndi.binary_dilation(bridge, iterations=3)
    pred[pred == b] = a                         # two neighbouring cells merged into one
    pred[bridge & (pred == 0)] = a
    big = ids[4]                                # one cell over-split
    ys, xs = np.nonzero(pred == big)
    pred[ys[xs > xs.mean()], xs[xs > xs.mean()]] = truth.max() + 1
    yy, xx = np.mgrid[:truth.shape[0], :truth.shape[1]]
    for k in range(2):                          # false detections on background
        for _ in range(100):
            cy, cx = rng.uniform(15, truth.shape[0] - 15, 2)
            blob = np.hypot(yy - cy, xx - cx) < 4
            if not (truth[blob] > 0).any():
                pred[blob] = truth.max() + 2 + k
                break
    return io.to_label_dtype(ndi.grey_dilation(pred, size=2) * (ndi.binary_dilation(pred > 0)))  # rough edges


def make_demo_project(root: Path, n_images: int = 3) -> Project:
    import tifffile

    project = Project(root)
    for i in range(n_images):
        img, truth, centres = _synthetic(seed=i)
        tifffile.imwrite(str(project.images_dir / f"demo_{i + 1:02d}.tif"), img)
        io.write_mask(project.predictions_dir / f"demo_{i + 1:02d}.tif", _imperfect(truth, centres, i))
    project.add_class("cells", "dividing cell")
    return project
