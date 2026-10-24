"""The improvement loop: measure your detector on corrected images, then make it better.

* evaluate()          – precision / recall / F1 of a detector against your corrected labels
* tune_classic()      – the built-in segmenter learns its settings from your corrections (CPU, minutes)
* finetune_cellpose() – re-train a Cellpose model on your corrections (CPU: slow but works; GPU: fast)
* write_notebooks()   – ready-made Google Colab notebooks to train on a FREE GPU
"""
from __future__ import annotations

import inspect
import json
import random
import time
from pathlib import Path

import numpy as np

from . import io, metrics
from .batch import single_channel
from .project import Project


def training_pairs(project: Project, label_set: str, only_done: bool = True, max_images: int | None = None,
                   seed: int = 0, channel: int = 0):
    """[(stem, single-channel image, corrected labels)] for images with saved corrections."""
    stems = [p.stem for p in project.images()
             if project.label_path(label_set, p.stem).exists() and (not only_done or project.status(p.stem) == "done")]
    if max_images and len(stems) > max_images:
        stems = sorted(random.Random(seed).sample(stems, max_images))
    out = []
    for stem in stems:
        image = io.read_image(project.image_path(stem))
        plane = single_channel(image, io.guess_layout(image), channel)
        labels, _, _ = project.load_labels(label_set, stem, plane.shape)
        out.append((stem, plane, labels))
    return out


def evaluate(project: Project, label_set: str, backend=None, params: dict | None = None,
             only_done: bool = True, record: bool = True, name: str | None = None, channel: int = 0) -> dict:
    """Compare a detector with your corrections. backend=None uses the saved detector output (predictions/)."""
    pairs = training_pairs(project, label_set, only_done, channel=channel)
    if not pairs:
        raise ValueError("No corrected images yet: save some images (and mark them Done) first.")
    per_image, results = {}, []
    for stem, plane, truth in pairs:
        if backend is None:
            try:
                pred, _ = project.prediction(label_set, stem, plane.shape)
            except ValueError:
                pred = None
            if pred is None:
                continue
        else:
            pred = backend.auto_segment(plane, params or {})
        r = metrics.match(truth, pred)
        results.append(r)
        per_image[stem] = metrics.public(r)
    if not results:
        raise ValueError("None of the corrected images has detector output to compare with.")
    overall = metrics.public(metrics.combine(results))
    entry = {"when": time.strftime("%Y-%m-%d %H:%M"), "label_set": label_set,
             "model": name or (backend.name if backend is not None else "saved detector output"),
             "images": len(results), **overall}
    if record:
        project.append_meta("evaluations", entry)
    return {"overall": entry, "per_image": per_image}


def _crop(a: np.ndarray, size: int = 768):
    if a.ndim != 2 or max(a.shape) <= size:
        return a
    y0 = max(0, (a.shape[0] - size) // 2)
    x0 = max(0, (a.shape[1] - size) // 2)
    return a[y0:y0 + size, x0:x0 + size]


def tune_classic(project: Project, label_set: str, n_trials: int = 30, max_images: int = 8, seed: int = 0,
                 channel: int = 0):
    """Generator: random search over the Classic segmenter's settings, scored by F1 on your corrections.

    Yields ("progress", i, n, best_f1) during the search and finally ("done", best_params, best_f1, old_f1).
    The project keeps the new settings only if they beat the current ones."""
    from .ai.classic import DEFAULT_PARAMS, ClassicBackend

    pairs = [(s, _crop(img), _crop(lab)) for s, img, lab in training_pairs(project, label_set, True, max_images, seed, channel)]
    pairs = [p for p in pairs if p[2].any()]
    if not pairs:
        raise ValueError("Tuning needs corrected images marked Done.")
    from .ai.base import polarity_from_labels

    current = {**DEFAULT_PARAMS, **(project.meta.get("classic_params") or {})}
    votes = [polarity_from_labels(img, lab) for _, img, lab in pairs]
    votes = [v for v in votes if v]
    polarity = max(set(votes), key=votes.count) if votes else current.get("polarity", "auto")
    sizes = [np.bincount(lab.ravel())[1:] for _, _, lab in pairs]
    med_area = float(np.median(np.concatenate([s[s > 0] for s in sizes]))) if sizes else 100.0

    def score(params):
        backend = ClassicBackend(params)
        return metrics.combine([metrics.match(lab, backend.auto_segment(img)) for _, img, lab in pairs])["f1"]

    rng = random.Random(seed)
    candidates = [current, {**DEFAULT_PARAMS, "polarity": polarity}]
    for _ in range(max(0, n_trials - 2)):
        candidates.append({
            "sigma": rng.choice([0.5, 0.8, 1.0, 1.5, 2.0, 3.0]),
            "threshold": rng.choice(["otsu", "li", "triangle", "mean"]),
            "offset": rng.choice([0.6, 0.8, 0.9, 1.0, 1.1, 1.25]),
            "polarity": polarity,                     # learned from your labels, not guessed
            "split": rng.choice([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]),
            "min_size": int(max(5, med_area * rng.choice([0.1, 0.2, 0.3, 0.5]))),
        })
    old_f1 = best_f1 = -1.0
    best = current
    for i, cand in enumerate(candidates):
        f1 = score(cand)
        if i == 0:
            old_f1 = f1
        if f1 > best_f1:
            best, best_f1 = cand, f1
        yield ("progress", i + 1, len(candidates), best_f1)
    if best_f1 > old_f1:
        project.set_meta("classic_params", best)
    yield ("done", best, best_f1, old_f1)


# ------------------------------------------------------------------ Cellpose fine-tuning
def finetune_cellpose(project: Project, label_set: str, base_model: str = "cyto3", epochs: int = 100,
                      name: str | None = None, channel: int = 0) -> Path:
    """Re-train Cellpose on your corrected images. Saves to models/cellpose/ so Ani lists it as a detector.

    CPU: roughly 5–30 minutes for 100 epochs on ~10 images; use the Colab notebook for big datasets."""
    from cellpose import models, train  # optional dependency

    from .ai.cellpose_backend import cellpose_major

    pairs = training_pairs(project, label_set, only_done=True, channel=channel)
    pairs = [p for p in pairs if p[1].ndim == 2 and len(np.unique(p[2])) > 1]
    if len(pairs) < 2:
        raise ValueError("Fine-tuning needs at least 2 corrected 2-D images marked Done.")
    rng = random.Random(0)
    rng.shuffle(pairs)
    n_test = max(1, len(pairs) // 5) if len(pairs) >= 5 else 0
    test, trn = pairs[:n_test], pairs[n_test:]
    save_dir = project.root / "models" / "cellpose"
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = project.root / "models" / "_cellpose_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)   # cellpose creates tmp_dir/models without parents=True
    name = name or f"{label_set}_{time.strftime('%Y%m%d-%H%M')}"
    if cellpose_major() >= 4:
        model = models.CellposeModel(gpu=False)
    else:
        model = models.CellposeModel(gpu=False, model_type=base_model)
    kwargs = dict(train_data=[p[1].astype(np.float32) for p in trn], train_labels=[p[2] for p in trn],
                  test_data=[p[1].astype(np.float32) for p in test] or None,
                  test_labels=[p[2] for p in test] or None,
                  channels=[0, 0], normalize=True, n_epochs=int(epochs), learning_rate=0.1 if cellpose_major() < 4 else 1e-5,
                  weight_decay=1e-4 if cellpose_major() < 4 else 0.1, SGD=cellpose_major() < 4,
                  save_path=str(tmp_dir), model_name=name, min_train_masks=1)
    accepted = inspect.signature(train.train_seg).parameters
    out = train.train_seg(model.net, **{k: v for k, v in kwargs.items() if k in accepted})
    path = Path(out[0] if isinstance(out, (tuple, list)) else out)
    final = save_dir / name
    if not path.exists():
        raise RuntimeError(f"Cellpose did not write a model file ({path})")
    path.replace(final)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    (save_dir / f"{name}.json").write_text(json.dumps({
        "base_model": base_model, "epochs": epochs, "images": len(trn), "label_set": label_set,
        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    return final


# ------------------------------------------------------------------ Colab notebooks (free GPU)
def _nb(cells: list[tuple[str, str]]) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [{"cell_type": t, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, src in cells]}


def write_notebooks(dataset_dir: Path, label_set: str = "cells") -> list[Path]:
    """Write Colab notebooks next to an exported dataset. Upload the zipped dataset folder to Colab and run."""
    dataset_dir = Path(dataset_dir)
    cellpose_nb = _nb([
        ("markdown", f"""# Train Cellpose on your Ani dataset (free GPU)
1. In Colab: **Runtime → Change runtime type → T4 GPU**.
2. Zip the exported dataset folder (`{dataset_dir.name}`) and upload it when the next cell asks.
3. Run all cells. Download `ani_cellpose_model` at the end and put it in your Ani project folder under
   `models/cellpose/`. Ani then lists it as a detector."""),
        ("code", """!pip -q install "cellpose>=3.1,<4"
from google.colab import files
up = files.upload()                      # choose your dataset .zip
zipname = next(iter(up))
!unzip -q -o "$zipname" -d data"""),
        ("code", f"""import glob, os, numpy as np, tifffile, imageio.v3 as iio
from cellpose import models, train, io as cpio
def load(split):
    imgs, labs = [], []
    for lab_path in sorted(glob.glob(f"data/**/masks/{{split}}/labels/{label_set}/*.tif", recursive=True)):
        stem = os.path.basename(lab_path)[:-4]
        img_path = [p for p in glob.glob(f"data/**/masks/{{split}}/images/{{stem}}.*", recursive=True)][0]
        img = tifffile.imread(img_path) if img_path.lower().endswith(("tif", "tiff")) else iio.imread(img_path)
        img = np.squeeze(img).astype(np.float32)
        if img.ndim == 3 and img.shape[-1] in (3, 4): img = img[..., :3].mean(-1)
        elif img.ndim == 3: img = img[0]
        imgs.append(img); labs.append(tifffile.imread(lab_path))
    return imgs, labs
X, Y = load("train"); Xv, Yv = load("val")
print(len(X), "training images,", len(Xv), "validation images")"""),
        ("code", """model = models.CellposeModel(gpu=True, model_type="cyto3")
path, tl, vl = train.train_seg(model.net, train_data=X, train_labels=Y,
                               test_data=Xv or None, test_labels=Yv or None,
                               channels=[0, 0], normalize=True, n_epochs=300, learning_rate=0.1,
                               weight_decay=1e-4, SGD=True, model_name="ani_cellpose_model",
                               save_path=".", min_train_masks=1)
print("saved", path)"""),
        ("code", """from cellpose import metrics as cpm
if Xv:
    m = models.CellposeModel(gpu=True, pretrained_model=str(path))
    pred = [m.eval(x, channels=[0, 0])[0] for x in Xv]
    ap = cpm.average_precision(Yv, pred)[0]
    print("validation AP@0.5:", ap[:, 0].mean().round(3))
files.download(str(path))"""),
    ])
    yolo_nb = _nb([
        ("markdown", f"""# Train YOLO-seg on your Ani dataset (free GPU)
1. **Runtime → Change runtime type → T4 GPU**.
2. Upload the zipped dataset folder (`{dataset_dir.name}`) when asked.
3. Run all cells, download `best.pt`, and put it in your Ani project under `models/yolo/`."""),
        ("code", """!pip -q install ultralytics
from google.colab import files
up = files.upload()
zipname = next(iter(up))
!unzip -q -o "$zipname" -d data"""),
        ("code", """import glob, yaml
cfg = glob.glob("data/**/yolo/data.yaml", recursive=True)[0]
d = yaml.safe_load(open(cfg)); d["path"] = cfg.rsplit("/", 1)[0]; yaml.safe_dump(d, open(cfg, "w"))
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")
results = model.train(data=cfg, epochs=100, imgsz=640, batch=8)"""),
        ("code", """best = glob.glob("runs/segment/*/weights/best.pt")[-1]
print(best)
files.download(best)"""),
    ])
    paths = []
    for fname, nb in (("train_cellpose_colab.ipynb", cellpose_nb), ("train_yolo_colab.ipynb", yolo_nb)):
        p = dataset_dir / fname
        p.write_text(json.dumps(nb, indent=1))
        paths.append(p)
    return paths
