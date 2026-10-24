"""Corrections and checks for outlines the user drew.

Brush strokes, hole filling and the review counts live here so they can be
tested without opening the window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from . import io


def brush_indices(shape, start, end, radius: float):
    """Pixels covered by a round brush dragged from start to end. A fast move still stays solid."""
    from skimage.draw import disk

    height, width = int(shape[0]), int(shape[1])
    radius = max(0.5, float(radius))
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    distance = float(np.hypot(*(end - start)))
    step = max(radius * 0.6, 1.0)
    count = max(1, int(np.ceil(distance / step)))
    rows, cols = [], []
    for index in range(count + 1):
        point = start + (end - start) * (index / count)
        rr, cc = disk((float(point[0]), float(point[1])), radius, shape=(height, width))
        if rr.size:
            rows.append(rr)
            cols.append(cc)
    if not rows:
        empty = np.zeros(0, np.int64)
        return empty, empty
    packed = np.unique(np.concatenate(rows).astype(np.int64) * width + np.concatenate(cols).astype(np.int64))
    return packed // width, packed % width


def paint_disk(labels: np.ndarray, row: float, col: float, radius: float, value: int) -> np.ndarray:
    """Paint a disk onto an outline mask. A value of 0 erases. The array is edited in place."""
    labels = np.asarray(labels)
    rr, cc = brush_indices(labels.shape, (row, col), (row, col), radius)
    if rr.size:
        labels[rr, cc] = int(value)
    return labels


def fill_label_holes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill background that is trapped inside cells with the cell around it."""
    labels = np.array(labels, copy=True)
    marked, count = ndimage.label(labels == 0)
    if count == 0:
        return labels, 0
    border = set(int(i) for i in np.unique(np.concatenate([
        marked[0].ravel(), marked[-1].ravel(), marked[:, 0].ravel(), marked[:, -1].ravel(),
    ])))
    filled = 0
    structure = np.ones((3, 3), bool)
    for index in range(1, int(count) + 1):
        if index in border:
            continue
        hole = marked == index
        ring = ndimage.binary_dilation(hole, structure=structure) & ~hole
        around = labels[ring]
        around = around[around > 0]
        if around.size == 0:
            continue
        labels[hole] = int(np.bincount(around.astype(np.int64)).argmax())
        filled += 1
    return labels, filled


def review_outlines(labels: np.ndarray, classes: dict | None = None) -> dict:
    """Count cells, holes, edge cuts, tiny fragments and cells with no class."""
    labels = np.asarray(labels)
    classes = {int(k): v for k, v in (classes or {}).items()}
    ids = [int(i) for i in np.unique(labels) if int(i) > 0]
    border = tiny = unclassed = 0
    for lab in ids:
        cell = labels == lab
        if int(cell.sum()) < 8:
            tiny += 1
        if cell[0].any() or cell[-1].any() or cell[:, 0].any() or cell[:, -1].any():
            border += 1
        if not str(classes.get(lab, "")).strip():
            unclassed += 1
    marked, count = ndimage.label(labels == 0)
    holes = 0
    if count:
        edge = set(int(i) for i in np.unique(np.concatenate([
            marked[0].ravel(), marked[-1].ravel(), marked[:, 0].ravel(), marked[:, -1].ravel(),
        ])))
        holes = sum(1 for i in range(1, int(count) + 1) if i not in edge)
    return {"cells": len(ids), "holes": holes, "border": border, "tiny": tiny, "unclassed": unclassed}


def review_message(report: dict) -> str:
    cells = int(report.get("cells", 0))
    if cells == 0:
        return "No cells yet."
    notes = []
    if report.get("unclassed"):
        notes.append(f"{report['unclassed']} without a class")
    if report.get("holes"):
        notes.append(f"{report['holes']} hole" + ("s" if report["holes"] != 1 else ""))
    if report.get("border"):
        notes.append(f"{report['border']} cut by the edge")
    if report.get("tiny"):
        notes.append(f"{report['tiny']} smaller than 8 pixels")
    noun = "cell" if cells == 1 else "cells"
    if not notes:
        return f"{cells} {noun}. Outlines are closed and classed."
    return f"{cells} {noun}. " + ", ".join(notes) + "."


def folder_progress(path: Path) -> tuple[int, int]:
    """How many images in this folder already have a saved seg.npy."""
    path = Path(path)
    images = io.list_images(path.parent)
    saved = sum(1 for image in images if image.with_name(f"{image.stem}_seg.npy").is_file())
    return saved, len(images)
