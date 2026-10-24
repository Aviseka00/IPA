"""Headless tests of the Ani panel (Linux CI: run under xvfb-run)."""

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from ani import io, tools  # noqa: E402
from ani.demo import _synthetic, make_demo_project  # noqa: E402
from ani.ui import AniWidget  # noqa: E402


def make(tmp_path, n=2):
    make_demo_project(tmp_path / "p", n_images=n)
    viewer = napari.Viewer(show=False)
    w = AniWidget(viewer, str(tmp_path / "p"))
    w.ask = lambda q: True
    w.sync_jobs = True
    viewer.window.add_dock_widget(w, name="Ani")
    return w


@pytest.fixture
def ani(tmp_path):
    w = make(tmp_path)
    yield w
    w._backup_timer.stop()
    w.viewer.close()


def n_obj(w):
    return len(tools.label_ids(w.labels_layer.data))


def test_correction_flow(ani, tmp_path):
    w = ani
    assert w.stem == "demo_01" and w.label_set == "cells" and w.tabs.count() == 4
    ids = list(tools.label_ids(w.labels_layer.data))
    n0 = len(ids)

    w.set_tool("Delete"); w.click_cell(ids[0])
    assert n_obj(w) == n0 - 1 and w.dirty
    w.set_tool("Merge"); w.click_cell(ids[1]); w.click_cell(ids[2])
    assert n_obj(w) == n0 - 2
    w.set_tool("Split"); w.click_cell(ids[1])
    assert n_obj(w) == n0 - 1

    w.undo(); assert n_obj(w) == n0 - 2
    w.undo(); assert n_obj(w) == n0 - 1
    w.redo(); assert n_obj(w) == n0 - 2

    w.set_tool("New cell")
    new_id = w.labels_layer.selected_label
    w.labels_layer.paint((5, 5), new_id)
    w.labels_layer.paint((6, 6), new_id)
    w._flush_paint()
    assert new_id in w.labels_layer.data
    w.undo(); assert new_id not in w.labels_layer.data
    w.redo(); assert new_id in w.labels_layer.data

    w.class_combo.setCurrentText("dividing cell")
    w.colour_by_class.setChecked(True)
    w.set_tool("Set class"); w.click_cell(ids[3])
    assert w.classes[ids[3]] == "dividing cell"
    w.undo(); assert ids[3] not in w.classes
    w.redo(); assert w.classes[ids[3]] == "dividing cell"

    w.set_tool("Outline")
    assert w.labels_layer.mode == "polygon"
    w.show_pred.setChecked(True)
    w.outlines_only.setChecked(True)
    assert any(lyr.name == "detector output" for lyr in w.viewer.layers)

    expected = n_obj(w)
    w.save_done_next()
    assert w.project.status("demo_01") == "done" and w.stem == "demo_02"
    w.load_image("demo_01")
    assert n_obj(w) == expected and w.classes[ids[3]] == "dividing cell" and not w.dirty

    w.with_notebooks.setChecked(True)
    w._export()
    out = next((tmp_path / "p" / "exports").glob("dataset-*"))
    assert list(out.glob("coco/*.json")) and (out / "train_cellpose_colab.ipynb").exists()


def test_find_problems_and_label_sets(ani):
    w = ani
    w.find_problems()
    assert len(w.problems) >= 4 and any(p["label"] == 0 for p in w.problems)
    assert w.problem_layer is not None and w.problem_i == 0
    w.goto_problem(1); assert w.problem_i == 1
    w.goto_problem(-1); assert w.problem_i == 0

    w.set_tool("Delete"); w.click_cell(int(tools.label_ids(w.labels_layer.data)[0]))
    key = w.project.add_label_set("nuclei")
    w.refresh_sets()
    w.set_combo.setCurrentText(key)
    assert w.label_set == "nuclei" and n_obj(w) == 0
    assert w.project.label_path("cells", "demo_01").exists()
    assert any(lyr.metadata.get("ani_other_set") for lyr in w.viewer.layers)
    w.labels_layer.paint((50, 50), 1)
    w._flush_paint()
    w.save()
    assert w.project.label_path("nuclei", "demo_01").exists()


def test_power_cut_recovery(ani):
    w = ani
    w.set_tool("Delete"); w.click_cell(int(tools.label_ids(w.labels_layer.data)[0]))
    n = n_obj(w)
    w.backup_now()
    assert w.project.has_recovery("cells", "demo_01")
    w.dirty = False
    w.stem = None
    w.load_image("demo_01", check_unsaved=False)
    assert n_obj(w) == n and w.dirty
    w.save()
    assert not w.project.has_recovery("cells", "demo_01")


def test_bad_files_do_not_crash(ani):
    w = ani
    (w.project.images_dir / "broken.tif").write_bytes(b"not an image")
    w.refresh_list()
    w.load_image("broken")
    assert "Could not open" in w.info.text() and w.stem == "demo_01"
    io.write_mask(w.project.predictions_dir / "demo_02.tif", np.ones((7, 7), np.uint16))
    w.load_image("demo_02")
    assert w.stem == "demo_02" and "EMPTY" in w.info.text()


# ------------------------------------------------------------------ AI tab
def test_ai_click_and_auto_label(ani):
    w = ani
    img, truth, _ = _synthetic(0)
    data = w.labels_layer.data
    missing = [i for i in tools.label_ids(truth) if not (data[truth == i] > 0).any()]
    n0 = n_obj(w)
    w.set_tool("AI click")
    pt = tuple(np.round(np.argwhere(truth == missing[0]).mean(0)).astype(int))
    w.ai_click(pt)
    assert n_obj(w) == n0 + 1
    new = w.labels_layer.data == w.labels_layer.data.max()
    assert (new & (truth == missing[0])).sum() / (new | (truth == missing[0])).sum() > 0.9
    w.undo(); assert n_obj(w) == n0
    w.ai_click((-5, 3))                                      # outside the image: ignored
    assert n_obj(w) == n0

    w.auto_label("add")                                      # keeps the user's objects, adds the missed ones
    assert n_obj(w) >= n0 + len(missing)
    w.auto_label("replace")
    from ani import metrics
    assert metrics.match(truth, w.labels_layer.data)["f1"] > 0.95
    w.undo(); w.undo()
    assert n_obj(w) == n0


def test_run_batch_and_rank(ani):
    w = ani
    w.overwrite.setChecked(True)
    w.run_batch()
    assert "finished" in w.batch_log.text()
    assert (w.project.predictions_dir / "cells" / "demo_02.tif").exists()
    w.rank_all()
    assert w.sort_combo.currentData() == "worst"
    texts = [w.image_list.item(i).text() for i in range(w.image_list.count())]
    assert len(texts) == 2
    w.use_command.setChecked(True)
    w.command_edit.setText("python detector.py")         # placeholders missing -> refused
    w.run_batch()
    assert "{input}" in w.info.text()


def test_train_tab(ani):
    w = ani
    w.evaluate()
    assert "corrected" in w.info.text().lower() or "no corrected" in w.info.text().lower()
    for path in w.project.images():
        img, truth, _ = _synthetic(int(path.stem[-2:]) - 1)
        w.project.save_labels("cells", path.stem, truth, {}, truth.shape)
        w.project.set_status(path.stem, "done")
    w.evaluate()
    assert w.eval_table.rowCount() == 1 and "F1" in w.train_log.text()
    w.eval_source.setCurrentIndex(1)                     # run the selected detector (Classic) now
    w.evaluate()
    assert w.eval_table.rowCount() == 2
    w.tune_trials.setValue(6)
    w.tune()
    assert "F1" in w.train_log.text()
    assert not w.ft_button.isEnabled()                   # cellpose not installed here


def test_job_guard(ani):
    w = ani
    w.jobs["local"] = object()                           # pretend something is running
    w.auto_label("replace")
    assert "still running" in w.info.text()
    w.jobs.clear()


def _wait(w, kind, seconds=60):
    import time

    from qtpy.QtWidgets import QApplication
    t0 = time.time()
    while kind in w.jobs and time.time() - t0 < seconds:
        QApplication.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        QApplication.processEvents()
    assert kind not in w.jobs, "background job did not finish"


def test_real_background_threads(ani):
    w = ani
    w.sync_jobs = False
    n0 = n_obj(w)
    w.auto_label("add")
    assert "local" in w.jobs                         # really running in a thread
    _wait(w, "local")
    assert n_obj(w) > n0
    w.overwrite.setChecked(True)
    w.run_batch()
    _wait(w, "batch")
    assert "finished" in w.batch_log.text()
    # stop a long batch early
    import tifffile
    img = io.read_image(w.project.images()[0])
    for i in range(12):
        tifffile.imwrite(str(w.project.images_dir / f"extra_{i:02d}.tif"), img)
    w.run_batch()
    w.stop_job("batch")
    _wait(w, "batch")
    done = len(list((w.project.predictions_dir / "cells").glob("extra_*.tif")))
    assert done < 12, "stop did not stop"
    # result for a different image is ignored
    w.auto_label("replace")
    w.load_image("demo_02", check_unsaved=False)
    _wait(w, "local")
    assert w.stem == "demo_02"


def test_errors_in_job_callbacks_are_shown(ani):
    w = ani

    def boom(_):
        raise RuntimeError("callback exploded")
    w.start_job("local", lambda: 1, on_done=boom)
    assert "callback exploded" in w.info.text() and "local" not in w.jobs


def test_fast_paths_background_save_and_layer_reuse(ani):
    """Speed features must never lose or show stale work."""
    w = ani
    w.sync_jobs = False
    img_layer, lab_layer = w._image_layers[0], w.labels_layer
    assert lab_layer._history_limit == 1                 # napari's unused undo history is off
    ids = list(tools.label_ids(w.labels_layer.data))
    w.set_tool("Delete")
    w.click_cell(ids[0])                                 # local change -> only changed pixels sent
    assert ids[0] not in w.labels_layer.data and (w.labels_layer.data == w._current[0]).all()
    expected = w._current[0].copy()
    # switch away and straight back: the background save must be on disk before the image is re-read
    w.step_image(1)
    assert w.stem == "demo_02"
    assert w._image_layers[0] is img_layer and w.labels_layer is lab_layer   # layers reused, not rebuilt
    w.step_image(-1)
    assert w.stem == "demo_01" and (w.labels_layer.data == expected).all()
    # undo / redo through the changed-pixels path
    w.click_cell(ids[1]); w.undo()
    assert (w.labels_layer.data == expected).all()
    w.redo(); assert ids[1] not in w.labels_layer.data
    # problems are found in the background
    w.find_problems()
    _wait(w, "local")
    assert w.problems and w.problem_layer is not None
    # quitting waits for queued saves
    w.save()
    w._on_quit()
    lab, _, src = w.project.load_labels("cells", "demo_01", w.image_shape)
    assert src == "your saved corrections" and ids[1] not in lab


def test_list_rows_update_without_rebuild(ani):
    w = ani
    first = w.image_list.item(0)
    w._set_status("flagged")
    assert w.image_list.item(0) is first and first.text().startswith("⚑")
