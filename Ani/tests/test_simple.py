"""The simple annotator: draw outlines, set a class, save seg.npy and training files."""

import json
from pathlib import Path

import numpy as np
import pytest

from ani.safety import atomic_write, clear_recovery, feature_job_error, load_recovery, save_recovery
from ani.seg import SAVE_KINDS, kind_from_filter, mask_from_glide, prepare_for_display, read_seg, write_segmentation


def test_a_failed_save_keeps_the_previous_file(tmp_path):
    target = tmp_path / "cells_seg.npy"
    target.write_text("previous", encoding="utf-8")

    def crash(path):
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        atomic_write(target, crash)
    assert target.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".*.partial*"))


def test_unsaved_cells_can_be_restored(tmp_path):
    image = tmp_path / "slide.tif"
    image.write_bytes(b"")
    masks = np.zeros((6, 8), np.uint16)
    masks[1:4, 2:5] = 1
    folder = tmp_path / "recovery"
    save_recovery(image, masks, ["cell", "infected"], {1: "infected"}, root=folder)
    loaded = load_recovery(image, root=folder)
    assert loaded is not None and loaded["classes"][1] == "infected"
    assert np.array_equal(loaded["masks"], masks)
    clear_recovery(image, root=folder)
    assert load_recovery(image, root=folder) is None


def test_recent_images_stay_on_this_computer(tmp_path):
    from ani.safety import remember_image, recent_images

    first = tmp_path / "a.tif"
    second = tmp_path / "b.tif"
    first.write_bytes(b"")
    second.write_bytes(b"")
    remember_image(first, root=tmp_path)
    remember_image(second, root=tmp_path)
    remember_image(first, root=tmp_path)
    assert [Path(p).name for p in recent_images(tmp_path)] == ["a.tif", "b.tif"]


def test_a_second_copy_sees_the_open_one(tmp_path, monkeypatch):
    from ani import safety

    monkeypatch.setattr(safety, "_process_alive", lambda pid: pid == 12345)
    (tmp_path / "ani.lock").write_text("12345", encoding="utf-8")
    assert safety.acquire_instance(tmp_path) is False
    (tmp_path / "ani.lock").write_text("999999", encoding="utf-8")
    assert safety.acquire_instance(tmp_path) is True


def test_feature_extraction_refuses_an_oversized_job():
    from ani.safety import MAX_FEATURE_CELLS, MAX_FEATURE_PIXELS

    assert feature_job_error(10, 10, 3) is None
    assert "pixels" in feature_job_error(1, MAX_FEATURE_PIXELS + 1, 1)
    assert "cells" in feature_job_error(10, 10, MAX_FEATURE_CELLS + 1)


def test_feature_table_records_shape_color_and_class(tmp_path):
    import csv

    from ani.features import FEATURE_COLUMNS, extract_cell_features, write_feature_table

    image = np.zeros((40, 50), np.uint8)
    image[5:15, 5:15] = 200
    image[20:30, 20:40] = 40
    labels = np.zeros((40, 50), np.uint16)
    labels[5:15, 5:15] = 1
    labels[20:30, 20:40] = 2
    rows = extract_cell_features(image, labels, {1: "infected", 2: "healthy"}, "slide.tif")
    assert [row["class"] for row in rows] == ["infected", "healthy"]
    assert rows[0]["area"] == 100 and rows[0]["intensity_mean"] == 200
    assert rows[1]["area"] == 200 and rows[1]["intensity_mean"] == 40
    assert rows[1]["bbox_width"] == 20 and rows[1]["aspect_ratio"] > rows[0]["aspect_ratio"]
    assert 0.5 < rows[0]["circularity"] <= 1
    assert rows[0]["mean_red"] == ""
    assert extract_cell_features(image, np.zeros_like(labels), {}) == []

    color = np.zeros((30, 30, 3), np.uint8)
    color[5:15, 5:15] = (180, 20, 10)
    painted = np.zeros((30, 30), np.uint16)
    painted[5:15, 5:15] = 1
    red = extract_cell_features(color, painted, {1: "dead"}, "color.png")[0]
    assert red["class"] == "dead" and red["mean_red"] == 180 and red["mean_green"] == 20
    assert red["intensity_mean"] > red["mean_blue"]

    table = write_feature_table(tmp_path / "cells_features", rows)
    with table.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert table.suffix == ".csv" and list(loaded[0]) == list(FEATURE_COLUMNS)
    assert loaded[0]["class"] == "infected" and float(loaded[0]["area"]) == 100


def test_training_bundle_can_rebuild_a_cell(tmp_path):
    import csv

    from ani.features import export_training_set, mask_from_rle, paste_cells, reconstruct_cell

    image = np.zeros((80, 90, 3), np.uint8)
    image[:] = (235, 225, 215)
    image[20:60, 18:58] = (150, 80, 40)
    image[32:48, 30:46] = (40, 40, 170)
    labels = np.zeros((80, 90), np.uint16)
    labels[20:60, 18:58] = 1
    labels[10:24, 64:80] = 2
    folder = tmp_path / "bundle"
    table = export_training_set(
        folder, image, labels, {1: "infected", 2: "uninfected"},
        image_name="slide.png", well="A1", slide="S3", magnification="20x",
    )
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    infected = next(row for row in rows if row["class"] == "infected")
    assert infected["well"] == "A1" and infected["slide"] == "S3" and infected["magnification"] == "20x"
    assert int(float(infected["nucleus_count"])) >= 1
    assert float(infected["nucleus_cell_area_ratio"]) > 0
    tight = mask_from_rle(infected["rle"], infected["rle_height"], infected["rle_width"])
    r0, c0 = int(infected["bbox_row"]), int(infected["bbox_col"])
    assert np.array_equal(tight, labels[r0:r0 + tight.shape[0], c0:c0 + tight.shape[1]])
    assert (folder / infected["patch"]).is_file()
    assert (folder / infected["mask_cell"]).is_file()
    assert (folder / infected["cnn_patch"]).is_file()
    assert (folder / "qc" / "slide_qc.png").is_file()
    assert (folder / "README.txt").is_file() and "class:" in (folder / "README.txt").read_text(encoding="utf-8")
    overlay = reconstruct_cell(folder, 1)
    assert overlay.ndim == 3 and overlay.shape[0] > 0
    pasted = paste_cells(folder)
    assert pasted.shape[:2] == image.shape[:2] and pasted[40, 38].sum() > 0 and pasted[0, 0].sum() == 0


def test_prepare_for_display_keeps_2d_and_colour():
    grey = np.zeros((20, 30), np.uint16)
    shown, info = prepare_for_display(grey)
    assert shown.shape == (20, 30) and info["note"] == ""

    rgb = np.zeros((20, 30, 3), np.uint8)
    shown, info = prepare_for_display(rgb)
    assert shown.shape == (20, 30, 3) and info["note"] == ""

    volume = np.zeros((7, 20, 30), np.uint16)
    volume[3, 4, 5] = 9
    shown, info = prepare_for_display(volume)
    assert shown.shape == (20, 30) and shown[4, 5] == 9 and info["plane"] == 3

    channels = np.zeros((3, 20, 30), np.uint16)
    channels[0, 1, 2] = 4
    shown, info = prepare_for_display(channels)
    assert shown.shape == (20, 30) and shown[1, 2] == 4 and info["channel"] == 0


def test_seg_roundtrip_and_training_files(tmp_path):
    masks = np.zeros((40, 50), np.uint16)
    masks[5:20, 6:22] = 1
    masks[22:35, 25:40] = 2
    out = tmp_path / "slide_seg.npy"
    written = write_segmentation(out, masks, ["cell", "nucleus"], {1: "cell", 2: "nucleus"}, image_name="slide.tif")
    names = {p.name for p in written}
    assert names == {"slide_seg.npy", "slide_classes.json", "slide_coco.json", "slide_yolo.txt"}

    loaded, classes, inst = read_seg(out)
    assert loaded.shape == masks.shape and np.array_equal(loaded, masks)
    assert classes == ["cell", "nucleus"] and inst == {1: "cell", 2: "nucleus"}

    meta = json.loads((tmp_path / "slide_classes.json").read_text(encoding="utf-8"))
    assert meta["instances"]["1"] == "cell" and meta["classes"] == ["cell", "nucleus"]
    coco = json.loads((tmp_path / "slide_coco.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in coco["categories"]] == ["cell", "nucleus"]
    assert len(coco["annotations"]) == 2
    assert coco["annotations"][0]["category_id"] == 1
    yolo = (tmp_path / "slide_yolo.txt").read_text(encoding="utf-8").splitlines()
    assert len(yolo) == 2 and yolo[0].startswith("0 ") and yolo[1].startswith("1 ")

    plain = tmp_path / "plain.npy"
    np.save(plain, masks)
    loaded, classes, inst = read_seg(plain)
    assert np.array_equal(loaded, masks) and classes == ["cell"] and inst == {}


def test_each_annotation_format_can_be_chosen(tmp_path):
    masks = np.zeros((30, 40), np.uint16)
    masks[4:16, 5:18] = 1
    labels = {"npy": ".npy", "tif": ".tif", "png": ".png", "coco": ".json", "yolo": ".txt"}
    assert [kind for kind, _ in SAVE_KINDS] == ["npy", "tif", "png", "coco", "yolo", "all"]
    assert kind_from_filter(SAVE_KINDS[0][1]) == "npy"
    assert kind_from_filter("Label mask TIFF (*.tif *.tiff)") == "tif"
    assert kind_from_filter("Label mask PNG (*.png)") == "png"
    assert kind_from_filter("COCO JSON (*.json)") == "coco"
    assert kind_from_filter("YOLO segmentation (*.txt)") == "yolo"
    assert kind_from_filter("All annotation formats (*.npy *.tif *.png *.json *.txt)") == "all"
    for kind, ext in labels.items():
        written = write_segmentation(tmp_path / f"cells_seg.npy", masks, ["cell"], {1: "cell"}, kind=kind)
        assert written[0].suffix.lower() == ext
        assert written[0].exists()
        assert not any(p.suffix.lower() not in {ext, ".json"} for p in written)


def test_glide_covers_the_whole_boundary():
    from scipy import ndimage as ndi

    theta = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    jitter = np.random.default_rng(0).normal(0, 0.35, (len(theta), 2))
    points = np.stack([40 + 16 * np.sin(theta), 50 + 16 * np.cos(theta)], 1) + jitter
    mask = mask_from_glide(points, (90, 110))
    assert mask is not None and mask[40, 50] and not mask[0, 0]
    boundary = mask & ~ndi.binary_erosion(mask)
    _, parts = ndi.label(boundary, structure=np.ones((3, 3), int))
    assert parts == 1 and boundary.sum() > 60
    assert mask_from_glide(np.array([[0, 0], [1, 1]]), (10, 10)) is None
    ring = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.stack([40 + 18 * np.sin(ring), 50 + 18 * np.cos(ring)], 1)
    held = mask_from_glide(circle, (90, 110))
    rr = np.clip(np.rint(circle[:, 0]).astype(int), 0, 89)
    cc = np.clip(np.rint(circle[:, 1]).astype(int), 0, 109)
    assert held is not None and held[rr, cc].mean() > 0.9
    ys, xs = np.nonzero(held)
    assert abs(float(ys.mean()) - 40) < 1.5 and abs(float(xs.mean()) - 50) < 1.5


def _ctrl_click(widget, position, canvas):
    event = type("E", (), {
        "type": "mouse_press", "button": 1, "modifiers": ("Control",),
        "position": position, "pos": np.asarray(canvas, float), "handled": False,
    })()
    gen = widget._on_glide(widget.labels_layer, event)
    next(gen)
    event.type = "mouse_release"
    try:
        next(gen)
    except StopIteration:
        pass
    return event


def test_ctrl_click_outlines_a_cell(tmp_path):
    napari = pytest.importorskip("napari")
    import tifffile

    from ani.simple_ui import SimpleAnnotator

    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, np.zeros((80, 90), np.uint16))
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        drag = type("E", (), {
            "type": "mouse_press", "button": 1, "modifiers": ("Control",),
            "position": (10, 10), "pos": np.array([0.0, 0.0]), "handled": False,
        })()
        gen = widget._on_glide(widget.labels_layer, drag)
        next(gen)
        drag.type = "mouse_move"
        drag.pos = np.array([40.0, 0.0])
        next(gen)
        drag.type = "mouse_release"
        try:
            next(gen)
        except StopIteration:
            pass
        assert widget._poly_pts == []

        _ctrl_click(widget, (20, 20), (0, 0))
        _ctrl_click(widget, (20, 50), (40, 0))
        _ctrl_click(widget, (50, 50), (80, 0))
        assert len(widget._poly_pts) == 3
        widget._on_mouse(widget.labels_layer, type("E", (), {"type": "mouse_press", "button": 2})())
        assert len(widget._poly_pts) == 2
        _ctrl_click(widget, (50, 50), (80, 30))
        assert _ctrl_click(widget, (22, 22), (0, 40)).handled
        assert widget._poly_pts == []
        assert widget.labels_layer.data[35, 35] != 0
        assert widget.classes[int(widget.labels_layer.data[35, 35])] == "cell"
    finally:
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def test_draw_class_undo_and_save(tmp_path):
    napari = pytest.importorskip("napari")
    import tifffile

    from ani.simple_ui import SimpleAnnotator
    from ani import tools

    image = np.zeros((100, 120), np.uint16)
    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, image)
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        assert widget.stem == "cells" and widget.labels_layer is not None
        from napari.utils.key_bindings import coerce_keybinding
        assert coerce_keybinding("Control-Z") in widget.labels_layer.keymap
        assert coerce_keybinding("G") in widget.labels_layer.keymap
        from napari.utils.action_manager import action_manager
        action_manager._get_repeatable_shortcuts(widget.labels_layer.keymap)
        assert not widget.labels_layer.mouse_pan
        drag = type("E", (), {"type": "mouse_press", "button": 1, "modifiers": (), "position": (5, 5), "pos": (5, 5), "handled": False})()
        gen = widget._on_glide(widget.labels_layer, drag)
        next(gen)
        assert drag.handled
        gen.close()
        widget._cancel_glide()
        widget.use_pan()
        assert widget.labels_layer.mouse_pan
        pan = type("E", (), {"type": "mouse_press", "button": 1, "modifiers": (), "position": (5, 5), "pos": (5, 5)})()
        try:
            next(widget._on_glide(widget.labels_layer, pan))
            took_the_drag = True
        except StopIteration:
            took_the_drag = False
        assert not took_the_drag and not getattr(pan, "handled", False)
        widget.use_glide()
        widget.add_class("nucleus")
        widget.labels_layer.paint_polygon([[10, 10], [10, 40], [40, 40], [40, 10]], 1)
        widget.flush()
        assert widget.classes[1] == "nucleus"
        widget.assign_class(1, "cell")
        assert widget.classes[1] == "cell"
        widget.labels_layer.paint_polygon([[50, 50], [50, 80], [90, 80], [90, 50]], 2)
        widget.flush()
        assert widget.classes[2] == "nucleus" and widget.dirty

        widget.undo()
        assert 2 not in tools.label_ids(widget.labels_layer.data)
        widget.redo()
        assert widget.classes[2] == "nucleus"
        table = widget.extract_to(tmp_path / "cells_features.csv")
        text = table[0].read_text(encoding="utf-8")
        assert "cell" in text and "nucleus" in text and "intensity_mean" in text

        called = {}
        widget._popup_class = lambda lab: called.__setitem__("lab", lab)
        event = type("E", (), {"type": "mouse_press", "button": 2, "position": (25, 25),
                                "view_direction": None, "dims_displayed": (0, 1)})()
        widget._on_mouse(widget.labels_layer, event)
        assert called["lab"] == 1

        written = widget.save_to(tmp_path / "cells_seg.npy")
        assert not widget.dirty and len(written) == 4
        loaded, classes, inst = read_seg(tmp_path / "cells_seg.npy")
        assert inst[1] == "cell" and inst[2] == "nucleus" and loaded.shape == (100, 120)

        widget.load_image(path)
        assert widget.classes[1] == "cell" and widget.classes[2] == "nucleus" and not widget.dirty
        assert widget.glide_btn.isChecked()
        theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        circle = np.stack([28 + 10 * np.sin(theta), 28 + 10 * np.cos(theta)], 1)
        widget._glide_pts = [circle[0], circle[1], circle[2]]
        widget._preview_glide()
        widget._clear_preview()
        assert widget.apply_glide(circle)
        assert widget.labels_layer.data[28, 28] != 0
    finally:
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def test_classic_segments_one_object_and_cellpose_is_optional(monkeypatch):
    from ani.ai.cellpose_backend import CellposeBackend
    from ani.segment import run_click, run_segment, segment_choices

    image = np.zeros((90, 90), np.uint8)
    image[15:55, 15:55] = 220
    labels = run_segment(image, "classic")
    assert labels[30, 30] > 0 and labels[0, 0] == 0
    first = int(labels[30, 30])
    image[60:82, 60:82] = 180
    mask, box = run_click(image, labels, (71, 71), "classic")
    assert mask is not None and mask.shape == labels[box].shape
    placed = np.array(labels, copy=True)
    crop = placed[box]
    crop[mask & (crop == 0)] = int(placed.max()) + 1
    placed[box] = crop
    assert placed[30, 30] == first and placed[71, 71] != 0 and placed[71, 71] != first
    colour = np.stack([image, image, image], axis=-1)
    assert run_segment(colour, "classic")[30, 30] > 0

    keys = [key for key, _label in segment_choices()]
    assert keys[0] == "classic"
    if not CellposeBackend("cyto3").available()[0]:
        assert "cellpose:cyto3" not in keys
    monkeypatch.setattr(CellposeBackend, "available", lambda self: (True, "ready"))
    keys = [key for key, _label in segment_choices()]
    assert "cellpose:cyto3" in keys and "cellpose:nuclei" in keys


def test_a_fast_brush_stroke_stays_connected():
    from ani.review import brush_indices

    labels = np.zeros((40, 40), np.uint16)
    rows, cols = brush_indices((40, 40), (20, 2), (20, 38), 2)
    labels[rows, cols] = 1
    assert labels[20, 2] == 1 and labels[20, 20] == 1 and labels[20, 38] == 1
    assert labels[0, 0] == 0


def test_corrections_scale_and_folder_progress(tmp_path):
    from ani.features import extract_cell_features
    from ani.review import fill_label_holes, folder_progress, paint_disk, review_message, review_outlines

    labels = np.zeros((24, 24), np.uint16)
    labels[4:20, 4:20] = 1
    labels[10:14, 10:14] = 0
    labels[0, 0] = 2
    report = review_outlines(labels, {1: "cell"})
    assert report["holes"] == 1 and report["tiny"] == 1 and report["unclassed"] == 1
    assert "hole" in review_message(report)
    filled, count = fill_label_holes(labels)
    assert count == 1 and filled[12, 12] == 1 and filled[0, 0] == 2
    paint_disk(filled, 12, 22, 2, 1)
    assert filled[12, 22] == 1
    paint_disk(filled, 12, 12, 1.5, 0)
    assert filled[12, 12] == 0

    image = np.full((20, 20), 10, np.uint8)
    mask = np.zeros((20, 20), np.uint16)
    mask[0:10, 0:10] = 1
    rows = extract_cell_features(image, mask, {1: "cell"}, pixel_size_um="0.5")
    assert rows[0]["area"] == 100 and float(rows[0]["area_um2"]) == 25
    assert float(rows[0]["pixel_size_um"]) == 0.5
    assert extract_cell_features(image, mask, {1: "cell"}, pixel_size_um="nope")[0]["area_um2"] == ""

    folder = tmp_path / "slide"
    folder.mkdir()
    (folder / "a.tif").write_bytes(b"")
    (folder / "b.png").write_bytes(b"")
    (folder / "a_seg.npy").write_bytes(b"")
    assert folder_progress(folder / "b.png") == (1, 2)


def test_brush_stroke_is_one_undo(tmp_path):
    napari = pytest.importorskip("napari")
    import tifffile
    from napari.utils.key_bindings import coerce_keybinding

    from ani.safety import clear_recovery
    from ani.simple_ui import SimpleAnnotator

    path = tmp_path / "brush.tif"
    tifffile.imwrite(path, np.zeros((40, 40), np.uint16))
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        widget.use_brush()
        assert widget.brush_btn.isChecked()
        assert coerce_keybinding("B") in widget.labels_layer.keymap
        assert coerce_keybinding("E") in widget.labels_layer.keymap
        event = type("E", (), {
            "type": "mouse_press", "button": 1, "modifiers": (),
            "position": (15, 15), "pos": np.array([15.0, 15.0]), "handled": False,
        })()
        gen = widget._on_glide(widget.labels_layer, event)
        next(gen)
        event.type = "mouse_move"
        event.position = (15, 30)
        next(gen)
        event.type = "mouse_release"
        try:
            next(gen)
        except StopIteration:
            pass
        assert int(widget.labels_layer.data.max()) == 1
        assert widget.classes[1] == "cell"
        widget.undo()
        assert int(widget.labels_layer.data.max()) == 0

        widget.labels_layer.data[8:16, 8:16] = 1
        widget.labels_layer.data[11:13, 11:13] = 0
        widget.flush()
        widget.fill_holes()
        assert widget.labels_layer.data[12, 12] == 1
    finally:
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def _wait_for_segment(widget):
    import time

    from qtpy.QtWidgets import QApplication

    deadline = time.time() + 20
    while widget._segment_job is not None and widget._segment_job.isRunning() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()


def test_segment_button_and_click_are_one_undo(tmp_path, monkeypatch):
    napari = pytest.importorskip("napari")
    import tifffile
    from napari.utils.key_bindings import coerce_keybinding
    from qtpy.QtWidgets import QMessageBox

    from ani.safety import clear_recovery
    from ani.simple_ui import SimpleAnnotator

    image = np.zeros((90, 90), np.uint8)
    image[15:55, 15:55] = 220
    image[60:82, 60:82] = 180
    path = tmp_path / "blobs.tif"
    tifffile.imwrite(path, image)
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        assert widget.segment_choice.currentData() == "spots"
        widget._select_segment_key("classic")
        assert widget.segment_choice.currentData() == "classic"
        assert coerce_keybinding("C") in widget.labels_layer.keymap
        assert widget.segment_image()
        _wait_for_segment(widget)
        assert int(widget.labels_layer.data[30, 30]) > 0
        assert "brush" in widget.status.text().lower()
        assert set(widget.classes.values()) == {"cell"}
        widget.undo()
        assert int(widget.labels_layer.data.max()) == 0

        widget.labels_layer.data[2, 2] = 1
        widget.flush()
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
        assert widget.segment_image() is False
        assert int(widget.labels_layer.data[2, 2]) == 1

        widget.labels_layer.data[2, 2] = 0
        widget.flush()
        widget.use_click()
        assert widget.click_btn.isChecked()
        assert widget.segment_at_point((30, 30))
        _wait_for_segment(widget)
        kept = int(widget.labels_layer.data[30, 30])
        assert kept > 0 and widget.classes[kept] == "cell"
        assert widget.segment_at_point((71, 71))
        _wait_for_segment(widget)
        other = int(widget.labels_layer.data[71, 71])
        assert other > 0 and other != kept and int(widget.labels_layer.data[30, 30]) == kept
        assert "brush" in widget.status.text().lower()
        widget.undo()
        assert int(widget.labels_layer.data[71, 71]) == 0
        assert int(widget.labels_layer.data[30, 30]) == kept
    finally:
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def test_this_pc_gets_a_worker_budget():
    from ani.machine import hardware

    info = hardware()
    assert info["logical"] >= 1
    assert info["workers"] >= 1
    assert info["workers"] <= info["logical"]
    assert info["feature_pixels"] >= 1_000_000


def test_lzw_tiff_opens(tmp_path):
    from PIL import Image

    from ani.io import read_image

    image = np.full((30, 40, 3), (190, 170, 120), np.uint8)
    path = tmp_path / "scan.tif"
    Image.fromarray(image).save(path, compression="tiff_lzw")
    loaded = read_image(path)
    assert loaded.shape == image.shape
    assert loaded.dtype == np.uint8


def test_infected_spots_mark_the_repeated_cell():
    from ani.spots import find_infected

    image = np.full((100, 130), 205, np.uint8)
    stamp = np.array(
        [[205, 205, 70, 205], [205, 35, 25, 205], [205, 90, 80, 205]],
        np.uint8,
    )
    image[40:43, 50:54] = stamp
    image[72:75, 51:55] = stamp
    other = np.array(
        [[205, 20, 205, 205], [20, 15, 20, 205], [205, 205, 90, 205]],
        np.uint8,
    )
    image[18:21, 100:104] = other
    labels, classes, meta = find_infected(image)
    assert labels.max() >= 2
    assert "infected" in classes.values()
    assert "duplicate" in classes.values()
    assert int((np.array(list(classes.values())) == "duplicate").sum()) == 1
    assert meta["kept"] >= 2
    assert meta["pairs"]
