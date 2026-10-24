"""An Ani project is just a folder, so students can copy/zip/share it easily.

    my_project/
      images/                  raw microscopy images (tif, png, jpg, bmp)
      predictions/             detector output for the FIRST label set, same name as the image
                               (img01.tif / img01_mask.png / img01.npy / img01.txt YOLO-seg)
      predictions/<set>/       detector output for other label sets (e.g. predictions/nuclei/img01.tif)
      labels/<set>/            human-corrected masks  (img01.tif + img01.classes.json)
      history/<set>/<image>/   last few saved versions (undo across sessions)
      .recovery/               automatic backups of unsaved work (power cut / crash)
      exports/                 training datasets made by "Export"
      project.json             label sets + classes, per-image status and correction statistics

A *label set* is one layer of objects that may overlap other sets, e.g. "cells", "nuclei",
"nucleoli", "mitochondria". Inside one set objects never overlap.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import io, tools

STATUSES = ("todo", "in progress", "flagged", "done")
HISTORY_KEEP = 5


def _safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name.strip())
    return keep.strip().replace(" ", "_")


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        for d in ("images", "predictions", "labels", "history", "exports", ".recovery"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()      # background jobs and the UI both update project.json
        self.meta_path = self.root / "project.json"
        self.meta = {"version": 2, "label_sets": {"cells": ["cell"]}, "images": {}}
        if self.meta_path.exists():
            self.meta.update(json.loads(self.meta_path.read_text()))
            self._migrate()

    def _migrate(self):
        """v0.1 projects: one flat class list, labels directly in labels/."""
        if "classes" in self.meta:
            self.meta["label_sets"] = {"cells": self.meta.pop("classes")}
            old = self.root / "labels"
            new = old / "cells"
            new.mkdir(exist_ok=True)
            for f in list(old.glob("*.tif")) + list(old.glob("*.classes.json")):
                f.replace(new / f.name)
            for info in self.meta["images"].values():
                st = info.get("stats")
                if st and "added" in st:
                    info["stats"] = {"cells": st}
            self.meta["version"] = 2
            self.save_meta()

    # ---------- folders ----------
    @property
    def images_dir(self): return self.root / "images"
    @property
    def predictions_dir(self): return self.root / "predictions"
    @property
    def labels_dir(self): return self.root / "labels"

    def save_meta(self):
        with self.lock:
            text = json.dumps(self.meta, indent=2)
            tmp = self.meta_path.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.replace(self.meta_path)

    def update_image(self, stem: str, key: str, label_set: str, value):
        """Thread-safe: meta["images"][stem][key][label_set] = value, then save."""
        with self.lock:
            self.info(stem).setdefault(key, {})[label_set] = value
            self.save_meta()

    # ---------- images ----------
    def images(self) -> list[Path]:
        """Images in images/, cached until the folder changes (listing thousands of files is slow)."""
        try:
            stamp = self.images_dir.stat().st_mtime_ns
        except OSError:
            stamp = None
        cache = getattr(self, "_img_cache", None)
        if cache is None or cache[0] != stamp or stamp is None:
            imgs = io.list_images(self.images_dir)
            self._img_cache = (stamp, imgs, {p.stem: p for p in imgs})
        return list(self._img_cache[1])

    def image_path(self, stem: str) -> Path | None:
        self.images()
        path = self._img_cache[2].get(stem)
        return path if path is not None and path.exists() else None

    def import_images(self, paths) -> int:
        n = 0
        for p in map(Path, paths):
            if p.suffix.lower() in io.IMAGE_EXTS:
                dst = self.images_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)
                    n += 1
        self._img_cache = None
        return n

    def info(self, stem: str) -> dict:
        with self.lock:
            return self.meta["images"].setdefault(stem, {"status": "todo"})

    def status(self, stem: str) -> str:
        return self.meta["images"].get(stem, {}).get("status", "todo")

    def set_status(self, stem: str, status: str):
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self.lock:
            self.info(stem)["status"] = status
            self.save_meta()

    # ---------- label sets & classes ----------
    @property
    def label_sets(self) -> list[str]:
        return list(self.meta["label_sets"])

    def classes(self, label_set: str) -> list[str]:
        return self.meta["label_sets"][label_set]

    def add_label_set(self, name: str, first_class: str | None = None) -> str:
        key = _safe_name(name)
        with self.lock:
            if key and key not in self.meta["label_sets"]:
                self.meta["label_sets"][key] = [first_class or name.strip()]
                self.save_meta()
        return key

    def add_class(self, label_set: str, name: str):
        name = name.strip()
        with self.lock:
            if name and name not in self.classes(label_set):
                self.classes(label_set).append(name)
                self.save_meta()

    def set_meta(self, key: str, value):
        with self.lock:
            self.meta[key] = value
            self.save_meta()

    def append_meta(self, key: str, value):
        with self.lock:
            self.meta.setdefault(key, []).append(value)
            self.save_meta()

    def categories(self) -> list[tuple[str, str, str]]:
        """All (label_set, class, export_name) – export names are unique across sets."""
        seen, out = {}, []
        for s in self.label_sets:
            for c in self.classes(s):
                seen[c] = seen.get(c, 0) + 1
        for s in self.label_sets:
            for c in self.classes(s):
                out.append((s, c, c if seen[c] == 1 else f"{s}/{c}"))
        return out

    # ---------- masks ----------
    def prediction_path(self, label_set: str, stem: str) -> Path | None:
        p = io.find_mask(self.predictions_dir / label_set, stem)
        if p is None and label_set == self.label_sets[0]:
            p = io.find_mask(self.predictions_dir, stem)
        return p

    def prediction(self, label_set: str, stem: str, shape) -> tuple[np.ndarray | None, dict[int, str]]:
        p = self.prediction_path(label_set, stem)
        if p is None:
            return None, {}
        mask, cls_idx = io.read_mask(p, shape)
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(f"detector mask {p.name} is {mask.shape} but the image is {tuple(shape)}")
        names = self.classes(label_set)
        return mask, {lab: names[i] if i < len(names) else f"class_{i}" for lab, i in cls_idx.items()}

    def label_path(self, label_set: str, stem: str) -> Path:
        return self.labels_dir / label_set / f"{stem}.tif"

    def _classes_path(self, folder: Path, stem: str) -> Path:
        return folder / f"{stem}.classes.json"

    @staticmethod
    def _read_classes(path: Path) -> dict[int, str]:
        return {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}

    @staticmethod
    def _write_classes(path: Path, classes: dict[int, str], labels: np.ndarray):
        present = set(map(int, tools.label_ids(labels)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in classes.items() if k in present}, indent=1))

    def has_labels(self, stem: str) -> bool:
        return any(self.label_path(s, stem).exists() for s in self.label_sets)

    def load_labels(self, label_set: str, stem: str, shape) -> tuple[np.ndarray, dict[int, str], str]:
        """Corrected labels if they exist, else the detector prediction, else empty.
        Returns (labels, {label: class}, source description)."""
        shape = tuple(shape)
        lp = self.label_path(label_set, stem)
        if lp.exists():
            mask, _ = io.read_mask(lp, shape)
            if mask.shape == shape:
                return mask, self._read_classes(self._classes_path(lp.parent, stem)), "your saved corrections"
            # never silently show something else: the user would think their corrections are lost
            return (np.zeros(shape, np.uint16), {},
                    f"EMPTY – your saved labels are {tuple(mask.shape)} but the image is {shape} "
                    f"(check 'Axis 0'); the file is kept unchanged unless you save")
        try:
            mask, classes = self.prediction(label_set, stem, shape)
        except Exception as e:  # unreadable or wrong size: start empty but tell the user
            return np.zeros(shape, np.uint16), {}, f"EMPTY – {e}"
        if mask is not None:
            return mask, classes, "detector output"
        return np.zeros(shape, np.uint16), {}, "empty (no detector output)"

    def save_labels(self, label_set: str, stem: str, labels: np.ndarray, classes: dict[int, str], shape) -> dict:
        lp = self.label_path(label_set, stem)
        if lp.exists():  # keep previous version in history
            hdir = self.root / "history" / label_set / stem
            hdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lp, hdir / f"{time.strftime('%Y%m%d-%H%M%S')}.tif")
            for old in sorted(hdir.glob("*.tif"))[:-HISTORY_KEEP]:
                old.unlink()
        io.write_mask(lp, labels)
        self._write_classes(self._classes_path(lp.parent, stem), classes, labels)
        self.clear_recovery(label_set, stem)
        try:
            pred, _ = self.prediction(label_set, stem, shape)
        except Exception:
            pred = None
        stats = tools.compare(pred, labels)
        with self.lock:
            info = self.info(stem)
            info.setdefault("stats", {})[label_set] = stats
            info["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if info.get("status") == "todo":
                info["status"] = "in progress"
            self.save_meta()
        return stats

    # ---------- crash / power-cut recovery ----------
    def recovery_path(self, label_set: str, stem: str) -> Path:
        return self.root / ".recovery" / label_set / f"{stem}.tif"

    def write_recovery(self, label_set: str, stem: str, labels: np.ndarray, classes: dict[int, str]):
        rp = self.recovery_path(label_set, stem)
        io.write_mask(rp, labels)
        self._write_classes(self._classes_path(rp.parent, stem), classes, labels)

    def has_recovery(self, label_set: str, stem: str) -> bool:
        rp, lp = self.recovery_path(label_set, stem), self.label_path(label_set, stem)
        return rp.exists() and (not lp.exists() or rp.stat().st_mtime > lp.stat().st_mtime)

    def load_recovery(self, label_set: str, stem: str, shape):
        rp = self.recovery_path(label_set, stem)
        mask, _ = io.read_mask(rp, shape)
        return mask, self._read_classes(self._classes_path(rp.parent, stem))

    def clear_recovery(self, label_set: str, stem: str):
        rp = self.recovery_path(label_set, stem)
        for f in (rp, self._classes_path(rp.parent, stem)):
            if f.exists():
                f.unlink()

    # ---------- summary ----------
    def summary(self) -> dict:
        counts = {s: 0 for s in STATUSES}
        totals = {"added": 0, "removed": 0, "modified": 0, "cells": 0}
        for img in self.images():
            counts[self.status(img.stem)] += 1
            for st in self.meta["images"].get(img.stem, {}).get("stats", {}).values():
                for k in totals:
                    totals[k] += st.get(k, 0)
        return {"status": counts, "corrections": totals}
