"""Load Cellpose-style *_seg.npy instance masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_instance_mask(seg_path: Path) -> np.ndarray:
    raw = np.load(seg_path, allow_pickle=True)
    data = raw.item() if getattr(raw, "ndim", None) == 0 else raw
    if not isinstance(data, dict) or "masks" not in data:
        raise ValueError(f"Unexpected seg format: {seg_path}")
    return np.asarray(data["masks"])


def instance_mask_to_binary(instance: np.ndarray) -> np.ndarray:
    return (instance > 0).astype(np.uint8)


def collect_instance_areas(instance: np.ndarray) -> list[int]:
    ids = [int(i) for i in np.unique(instance) if i > 0]
    return [int(np.sum(instance == i)) for i in ids]
