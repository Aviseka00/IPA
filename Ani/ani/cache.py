"""Small in-memory image cache so folder browsing feels instant."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np


class ImageCache:
    """Keep a few decoded images. Newest stays; oldest drops when full."""

    def __init__(self, max_items: int = 3):
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def _key(self, path) -> str:
        return str(Path(path).resolve())

    def get(self, path) -> np.ndarray | None:
        key = self._key(path)
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, path, image: np.ndarray) -> None:
        key = self._key(path)
        self._items[key] = np.asarray(image)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def drop(self, path) -> None:
        self._items.pop(self._key(path), None)

    def clear(self) -> None:
        self._items.clear()


IMAGE_CACHE = ImageCache(max_items=3)
