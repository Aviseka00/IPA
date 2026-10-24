"""Start Ani:  python -m ani [IMAGE] [--demo] [--selftest] [--download-models]"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def selftest(check_ai: bool = True) -> int:
    """Headless check that everything students need works on this computer. Returns 0 if all good."""
    import numpy as np

    from . import __version__, ai, io, metrics, tools
    from .demo import _imperfect, _synthetic, make_demo_project
    from .export import export_dataset
    from .train import evaluate, tune_classic, write_notebooks

    results = []

    def check(name, fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            results.append((True, name, f"{detail} ({time.time() - t0:.1f}s)"))
        except Exception as e:  # report and continue
            results.append((False, name, f"{type(e).__name__}: {e}"))

    print(f"Ani {__version__} self-test – Python {sys.version.split()[0]}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "selftest"
        state = {}

        def demo():
            state["p"] = make_demo_project(root, n_images=3)
            return f"{len(state['p'].images())} images"

        def classic():
            img, truth, _ = _synthetic(1)
            r = metrics.match(truth, ai.ClassicBackend().auto_segment(img))
            assert r["f1"] > 0.9, f"F1 only {r['f1']}"
            return f"F1 {r['f1']:.2f}"

        def wand():
            img, truth, c = _synthetic(0)
            pred = _imperfect(truth, c, 0)
            missing = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()][0]
            pt = np.round(np.argwhere(truth == missing).mean(0)).astype(int)
            mask, box = ai.ClassicBackend().segment_at(img, pt, pred)
            full = np.zeros(truth.shape, bool)
            full[box] = mask
            iou = (full & (truth == missing)).sum() / (full | (truth == missing)).sum()
            assert iou > 0.8, f"IoU only {iou:.2f}"
            return f"IoU {iou:.2f}"

        def finder():
            img, truth, c = _synthetic(0)
            n = len(tools.find_suspects(_imperfect(truth, c, 0))) + len(tools.find_missed(img, _imperfect(truth, c, 0)))
            assert n >= 4
            return f"{n} problems found"

        def save_export():
            p = state["p"]
            for path in p.images():
                image = io.read_image(path)
                lab, _, _ = p.load_labels("cells", path.stem, image.shape)
                p.save_labels("cells", path.stem, lab, {}, image.shape)
                p.set_status(path.stem, "done")
            out = export_dataset(p, Path(tmp) / "export")
            write_notebooks(out)
            return "masks + COCO + YOLO + notebooks"

        def train_loop():
            p = state["p"]
            r = evaluate(p, "cells")["overall"]
            final = list(tune_classic(p, "cells", n_trials=4))[-1]
            return f"detector F1 {r['f1']:.2f}, tuned Classic F1 {final[2]:.2f}"

        def gui():
            import napari  # noqa: F401
            from qtpy import QtCore  # noqa: F401
            return f"napari {napari.__version__}"

        def annotator():
            from .features import extract_cell_features
            from .review import brush_indices, fill_label_holes
            from .seg import read_seg, write_segmentation

            labels = np.zeros((32, 32), np.uint16)
            rr, cc = brush_indices(labels.shape, (8, 4), (8, 28), 2)
            labels[rr, cc] = 1
            assert labels[8, 16] == 1
            block = np.zeros((32, 32), np.uint16)
            block[2:30, 2:30] = 1
            block[14:18, 14:18] = 0
            filled, holes = fill_label_holes(block)
            assert holes == 1 and filled[16, 16] == 1
            image = np.full((32, 32), 9, np.uint8)
            rows = extract_cell_features(image, filled, {1: "object"})
            assert rows and rows[0]["class"] == "object"
            path = Path(tmp) / "annotator_seg.npy"
            write_segmentation(path, filled, ["object"], {1: "object"}, kind="npy")
            loaded, _names, classes = read_seg(path)
            assert classes[1] == "object" and loaded.shape == filled.shape
            return "brush, holes, features, seg.npy"

        check("annotator", annotator)
        check("demo project", demo)
        check("built-in detector", classic)
        check("AI click (magic wand)", wand)
        check("problem finder", finder)
        check("save + export", save_export)
        check("evaluate + tune", train_loop)
        check("viewer libraries", gui)
        if check_ai:
            for b in ai.backends_for(None)[1:]:
                ok, msg = b.available()
                results.append((None, b.name, msg))
    for ok, name, detail in results:
        mark = {True: "PASS", False: "FAIL", None: "info"}[ok]
        print(f"  [{mark}] {name}: {detail}")
    failed = [r for r in results if r[0] is False]
    print("All checks passed." if not failed else f"{len(failed)} check(s) failed.")
    return 1 if failed else 0


def download_models() -> int:
    """Fetch the weights of every installed AI package once, so Ani works offline afterwards."""
    import importlib.util

    steps = []
    if importlib.util.find_spec("cellpose"):
        def cellpose_models():
            from .ai.cellpose_backend import CellposeBackend
            from .demo import _synthetic
            img = _synthetic(0, size=128, n=6)[0]
            for kind in ("cyto3", "nuclei"):
                CellposeBackend(kind).auto_segment(img)      # first use downloads the weights
            return "cyto3 + nuclei ready"
        steps.append(("Cellpose", cellpose_models))
    if importlib.util.find_spec("micro_sam"):
        def sam_model():
            from micro_sam import util
            util.get_sam_model(model_type="vit_t_lm", device="cpu")
            return f"vit_t_lm ready (cache: {util.microsam_cachedir()})"
        steps.append(("micro-sam", sam_model))
    if importlib.util.find_spec("ultralytics"):
        def yolo():
            import ultralytics
            return (f"ultralytics {ultralytics.__version__} ready – train a model with the Colab notebook from "
                    f"Ani's export and put best.pt in <project>/models/yolo/")
        steps.append(("YOLO", yolo))
    if not steps:
        print("No AI packages installed. Run install_ai.bat first.")
        return 1
    failed = 0
    for name, fn in steps:
        print(f"{name}: downloading / checking…", flush=True)
        try:
            print(f"  [OK] {fn()}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {type(e).__name__}: {e}")
    return 1 if failed else 0


def _startup_image(args) -> Path | None:
    """Image to show when the simple annotator opens."""
    from . import io
    from .demo import make_demo_project

    if args.smoke:
        root = Path(tempfile.mkdtemp()) / "smoke"
        make_demo_project(root, n_images=1)
        return next((root / "images").glob("demo_01.*"))
    if args.demo:
        root = Path(args.project or Path.home() / "Ani-demo")
        if not (root / "project.json").exists():
            make_demo_project(root)
        images = io.list_images(root / "images")
        return images[0] if images else None
    if not args.project:
        return None
    path = Path(args.project)
    if path.is_file():
        return path
    if path.is_dir():
        folder = path / "images" if (path / "images").is_dir() else path
        images = io.list_images(folder)
        return images[0] if images else None
    return None


def _run_smoke(viewer, ok) -> None:
    from qtpy.QtCore import QTimer

    box = {}

    def finish():
        box["ok"] = bool(ok())
        print("SMOKE OK" if box["ok"] else "SMOKE FAILED", flush=True)
        viewer.close()

    QTimer.singleShot(3000, finish)
    import napari
    napari.run()
    sys.exit(0 if box.get("ok") else 1)


def _run_annotator(args):
    import napari

    from .simple_ui import SimpleAnnotator

    viewer = napari.Viewer(title="Ani")
    widget = SimpleAnnotator(viewer)
    dock = viewer.window.add_dock_widget(widget, name="Ani", area="right")
    try:
        dock.setMinimumWidth(widget.minimumWidth())
    except Exception:
        pass
    image = _startup_image(args)
    if image is not None:
        widget.load_image(image)
    if args.smoke:
        _run_smoke(viewer, lambda: widget.labels_layer is not None and widget.stem is not None)
    napari.run()


def _run_workspace(args):
    """The previous project workspace: batches, evaluation and training."""
    import napari

    from .ui import AniWidget

    project = args.project
    if args.demo:
        from .demo import make_demo_project

        project = project or str(Path.home() / "Ani-demo")
        if not (Path(project) / "project.json").exists():
            make_demo_project(Path(project))
    if args.smoke:
        project = project or str(Path(tempfile.mkdtemp()) / "smoke")
        if not (Path(project) / "project.json").exists():
            from .demo import make_demo_project
            make_demo_project(Path(project), n_images=1)
    viewer = napari.Viewer(title="Ani")
    widget = AniWidget(viewer, project)
    viewer.window.add_dock_widget(widget, name="Ani", area="right")
    if args.smoke:
        _run_smoke(viewer, lambda: widget.labels_layer is not None and widget.stem is not None)
    napari.run()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ani", description="Ani – outline objects in any image and save them for training")
    ap.add_argument("project", nargs="?", help="image to open, or a folder of images")
    ap.add_argument("--demo", action="store_true", help="open a sample image you can practise on")
    ap.add_argument("--full", action="store_true", help="open the full project workspace")
    ap.add_argument("--selftest", action="store_true", help="check that Ani works on this computer, then exit")
    ap.add_argument("--download-models", action="store_true",
                    help="download the weights of installed AI packages (Cellpose, micro-sam), then exit")
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)  # open the window, close after 3 s
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(selftest())
    if args.download_models:
        sys.exit(download_models())
    from .safety import acquire_instance, install_excepthook, install_logging
    install_excepthook()
    install_logging()
    if not args.smoke and not acquire_instance():
        print("Ani is already open on this computer.", flush=True)
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, "Ani is already open on this computer.", "Ani", 0x40)
            except Exception:
                pass
        return
    if args.full:
        _run_workspace(args)
    else:
        _run_annotator(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # needed by the Windows .exe build
    main()
