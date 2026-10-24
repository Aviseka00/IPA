import json

import numpy as np

from ani import io, tools
from ani.demo import _imperfect, _synthetic, make_demo_project
from ani.export import export_dataset, mask_to_polygons
from ani.project import Project


def two_cells():
    m = np.zeros((40, 60), np.uint16)
    m[10:30, 5:25] = 1
    m[10:30, 27:50] = 2
    return m


def test_delete_merge_split():
    m = two_cells()
    assert set(tools.label_ids(tools.delete_label(m, 2))) == {1}
    merged = tools.merge_labels(m, 1, 2)
    assert set(tools.label_ids(merged)) == {1}
    assert (merged[15, 25:27] == 1).all()  # gap filled
    split = tools.split_label(merged, 1, 2)
    assert len(tools.label_ids(split)) == 2
    assert (split > 0).sum() == (merged > 0).sum()


def test_split_round_blob_and_3d():
    yy, xx = np.mgrid[:50, :50]
    m = (np.hypot(yy - 25, xx - 25) < 15).astype(np.uint16)
    assert len(tools.label_ids(tools.split_label(m, 1, 3))) == 3
    vol = np.stack([two_cells()] * 5)
    assert len(tools.label_ids(tools.merge_labels(vol, 1, 2))) == 1


def test_cleanup_and_relabel():
    m = two_cells()
    m[0, 0] = 7
    m[15:18, 10:13] = 0  # hole
    m = tools.fill_holes(tools.remove_small(m, 5))
    assert 7 not in m and m[16, 11] == 1
    m[m == 2] = 9
    out, cls = tools.relabel_sequential(m, {9: "dividing cell"})
    assert set(tools.label_ids(out)) == {1, 2} and cls == {2: "dividing cell"}


def test_compare():
    gt = two_cells()
    pred = gt.copy()
    pred[pred == 2] = 0
    pred[0:5, 0:5] = 5  # false detection
    s = tools.compare(pred, gt)
    assert s == {"cells": 2, "matched": 1, "added": 1, "removed": 1, "modified": 0}


def test_problem_finder_on_demo_errors():
    img, truth, centres = _synthetic(seed=0)
    pred = _imperfect(truth, centres, 0)
    suspects = tools.find_suspects(pred)
    reasons = " ".join(s["reason"] for s in suspects)
    assert "merged" in reasons and ("small" in reasons or "pieces" in reasons)
    missed = tools.find_missed(img, pred)
    # the two cells deliberately removed from the prediction must be found
    missing_ids = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()]
    hits = 0
    for i in missing_ids:
        cy, cx = np.argwhere(truth == i).mean(0)
        hits += any(np.hypot(cy - m["centroid"][0], cx - m["centroid"][1]) < 10 for m in missed)
    assert hits == len(missing_ids) >= 2
    assert tools.find_missed(img, truth) == []  # nothing missed in the perfect labels


def test_yolo_roundtrip(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n")
    m, c = io.read_mask(t, (100, 100))
    assert len(tools.label_ids(m)) == 2 and c == {1: 0, 2: 1}


def test_polygon_offsets():
    m = two_cells()
    poly = mask_to_polygons(m, 2)[0]
    assert abs(poly[:, 0].min() - 27) <= 1 and abs(poly[:, 1].min() - 10) <= 1


def test_project_sets_recovery_export(tmp_path):
    p = make_demo_project(tmp_path / "proj", n_images=3)
    key = p.add_label_set("nuclei")
    p.add_class("cells", "cell")  # duplicate ignored
    assert p.label_sets == ["cells", "nuclei"] and p.classes("nuclei") == ["nuclei"]
    imgs = p.images()
    for img in imgs:
        image = io.read_image(img)
        lab, cls, src = p.load_labels("cells", img.stem, image.shape)
        assert src == "detector output" and lab.max() > 0
        first = int(tools.label_ids(lab)[0])
        stats = p.save_labels("cells", img.stem, lab, {first: "dividing cell"}, image.shape)
        assert stats["added"] == 0
        nuc = (lab > 0).astype(np.uint16) * 0
        nuc[10:20, 10:20] = 1
        p.save_labels(key, img.stem, nuc, {}, image.shape)
        p.set_status(img.stem, "done")
    # recovery
    stem = imgs[0].stem
    assert not p.has_recovery("cells", stem)
    p.write_recovery("cells", stem, np.zeros(image.shape, np.uint16), {})
    assert p.has_recovery("cells", stem)
    p.clear_recovery("cells", stem)
    assert not p.has_recovery("cells", stem)
    # wrong-size prediction does not crash, it loads empty with a warning
    io.write_mask(p.predictions_dir / "nuclei" / "demo_02.tif", np.ones((10, 10), np.uint16))
    (p.labels_dir / "nuclei" / "demo_02.tif").unlink()
    lab, _, src = p.load_labels("nuclei", "demo_02", image.shape)
    assert src.startswith("EMPTY") and lab.shape == image.shape and lab.max() == 0

    out = export_dataset(p, tmp_path / "ds")
    coco = json.loads((out / "coco" / "train.json").read_text())
    names = {c["name"] for c in coco["categories"]}
    assert names == {"cell", "dividing cell", "nuclei"}
    assert {a["category_id"] for a in coco["annotations"]} >= {1, 2, 3}
    assert (out / "yolo" / "data.yaml").exists()
    png = next((out / "yolo" / "images").rglob("*.png"))
    assert io.read_image(png).dtype == np.uint8
    assert (out / "masks" / "train" / "labels" / "nuclei").exists()
    assert p.summary()["status"]["done"] == 3


def test_migrate_v1_project(tmp_path):
    root = tmp_path / "old"
    (root / "labels").mkdir(parents=True)
    io.write_mask(root / "labels" / "a.tif", two_cells())
    (root / "project.json").write_text(json.dumps({"classes": ["cell", "x"], "images": {
        "a": {"status": "done", "stats": {"cells": 2, "added": 0, "removed": 0, "modified": 0, "matched": 2}}}}))
    p = Project(root)
    assert p.label_sets == ["cells"] and p.classes("cells") == ["cell", "x"]
    assert p.label_path("cells", "a").exists()
    assert p.meta["images"]["a"]["stats"]["cells"]["cells"] == 2


def test_saved_labels_with_wrong_shape_are_not_hidden(tmp_path):
    p = make_demo_project(tmp_path / "p", n_images=1)
    io.write_mask(p.label_path("cells", "demo_01"), np.ones((5, 5), np.uint16))
    lab, _, src = p.load_labels("cells", "demo_01", (384, 384))
    assert src.startswith("EMPTY") and "Axis 0" in src and lab.max() == 0
    assert p.label_path("cells", "demo_01").exists()                     # file untouched


def test_polygons_ignore_holes():
    m = np.zeros((40, 40), np.uint16)
    m[5:35, 5:35] = 1
    m[15:25, 15:25] = 0          # a hole
    polys = mask_to_polygons(m, 1)
    assert len(polys) == 1
