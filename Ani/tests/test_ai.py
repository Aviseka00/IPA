"""Phase 3–5 tests: built-in detector, AI adapters (with fake libraries), batch runs, evaluation and training."""
import ast
import importlib
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from ani import ai, batch, io, metrics, tools, train
from ani.demo import _imperfect, _synthetic, make_demo_project

FAKES = str(Path(__file__).parent / "fakes")


@pytest.fixture
def fakes(monkeypatch):
    """Make the fake cellpose / micro_sam / ultralytics importable for one test."""
    monkeypatch.syspath_prepend(FAKES)
    for name in list(sys.modules):
        if name.split(".")[0] in ("cellpose", "micro_sam", "ultralytics", "mobile_sam"):
            monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    yield
    for name in list(sys.modules):
        if name.split(".")[0] in ("cellpose", "micro_sam", "ultralytics", "mobile_sam"):
            del sys.modules[name]


def dense(seed=0, H=384, n=150, r=11):
    from skimage.segmentation import relabel_sequential
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, H, (n, 2)).astype(int)
    seeds = np.zeros((H, H), np.int32)
    seeds[pts[:, 0], pts[:, 1]] = np.arange(1, n + 1)
    dist, (iy, ix) = ndi.distance_transform_edt(seeds == 0, return_indices=True)
    lab = np.where(dist < r, seeds[iy, ix], 0)
    lab = lab * ~(ndi.grey_dilation(lab, 3) != ndi.grey_erosion(np.where(lab == 0, 10 ** 6, lab), 3))
    lab = relabel_sequential(lab)[0].astype(np.uint16)
    inner = ndi.distance_transform_edt(lab > 0)
    im = ndi.gaussian_filter((lab > 0) * (0.6 + 0.4 * np.minimum(inner / 6, 1)), 1.5) * 2000 + 300
    im = (im + rng.normal(0, 40, lab.shape)).astype(np.float32)
    return im, lab


# ------------------------------------------------------------------ metrics
def test_metrics_known_values():
    gt = np.zeros((20, 20), np.uint16)
    gt[2:8, 2:8] = 1
    gt[10:16, 10:16] = 2
    pred = np.zeros_like(gt)
    pred[2:8, 2:8] = 5           # perfect
    pred[0:2, 15:20] = 6         # false positive
    r = metrics.match(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5 and r["mean_iou"] == 1.0
    assert metrics.match(gt, gt)["f1"] == 1.0
    empty = np.zeros_like(gt)
    assert metrics.match(empty, empty)["f1"] == 1.0
    assert metrics.match(gt, empty)["recall"] == 0.0
    c = metrics.combine([metrics.match(gt, pred), metrics.match(gt, gt)])
    assert (c["tp"], c["fp"], c["fn"]) == (3, 1, 1)
    with pytest.raises(ValueError):
        metrics.match(gt, np.zeros((5, 5)))


# ------------------------------------------------------------------ classic backend
def test_classic_fluorescence_and_brightfield():
    cb = ai.ClassicBackend()
    for seed in range(3):
        img, truth, _ = _synthetic(seed)
        assert metrics.match(truth, cb.auto_segment(img))["f1"] >= 0.95
        bright_field = (65535 - img).astype(np.uint16)
        assert metrics.match(truth, cb.auto_segment(bright_field))["f1"] >= 0.95


def test_classic_3d_and_flat_images():
    img, truth, _ = _synthetic(0, size=128, n=8)
    vol = np.stack([img] * 4)
    lab = ai.ClassicBackend().auto_segment(vol)
    assert lab.shape == vol.shape and lab.max() >= 6
    assert ai.ClassicBackend().auto_segment(np.zeros((50, 50))).max() == 0          # flat image, no crash
    assert ai.ClassicBackend().auto_segment(np.full((50, 50), 7.0)).max() == 0


def test_magic_wand_accuracy():
    cb = ai.ClassicBackend()
    img, truth, c = _synthetic(0)
    pred = _imperfect(truth, c, 0)
    missing = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()]
    for i in missing:
        pt = np.round(np.argwhere(truth == i).mean(0)).astype(int)
        mask, box = cb.segment_at(img, pt, pred)
        full = np.zeros(truth.shape, bool)
        full[box] = mask
        assert (full & (truth == i)).sum() / (full | (truth == i)).sum() > 0.9
        assert not (full & (pred > 0)).any()                     # never paints over existing labels
    inside = tuple(np.argwhere(pred > 0)[0])
    assert cb.segment_at(img, inside, pred) == (None, None)     # click on a labelled pixel
    # dense tissue
    im, lab = dense()
    ious = []
    for t in range(20, 35):
        p = lab.copy()
        p[p == t] = 0
        pt = np.round(np.argwhere(lab == t).mean(0)).astype(int)
        if lab[tuple(pt)] != t:
            continue
        mask, box = cb.segment_at(im, pt, p)
        full = np.zeros(lab.shape, bool)
        if mask is not None:
            full[box] = mask
        ious.append((full & (lab == t)).sum() / max(1, (full | (lab == t)).sum()))
    assert np.median(ious) > 0.8


def test_add_new_objects():
    old = np.zeros((30, 30), np.uint16)
    old[2:10, 2:10] = 4
    new = np.zeros_like(old)
    new[3:11, 3:11] = 1      # same cell (big overlap) -> ignored
    new[20:26, 20:26] = 2    # new cell -> added
    out = tools.add_new_objects(old, new)
    assert set(tools.label_ids(out)) == {4, 5}
    assert (out[2:10, 2:10] == 4).all() and (out[20:26, 20:26] == 5).all()
    assert (tools.add_new_objects(old, np.zeros_like(old)) == old).all()


def test_problem_reasons_are_readable():
    img, truth, c = _synthetic(0)
    s = tools.find_suspects(_imperfect(truth, c, 0))
    assert all(p["reason"] and "{" not in p["reason"] for p in s)


# ------------------------------------------------------------------ AI adapters (fake libraries)
def test_cellpose_adapter(fakes, tmp_path):
    import cellpose
    b = ai.CellposeBackend("nuclei")
    assert b.available()[0]
    img, truth, _ = _synthetic(1)
    lab = b.auto_segment(img)
    assert metrics.match(truth, lab)["f1"] > 0.9
    assert ("Cellpose", {"model_type": "nuclei", "gpu": False}) in cellpose.CALLS
    ev = [kw for name, kw in cellpose.CALLS if name == "CellposeModel.eval"][-1]
    assert ev["channels"] == [0, 0] and ev["diameter"] is None
    b.auto_segment(img, {"diameter": 25})
    assert [kw for n, kw in cellpose.CALLS if n == "CellposeModel.eval"][-1]["diameter"] == 25
    b.auto_segment(np.stack([img, img]))
    ev = [kw for n, kw in cellpose.CALLS if n == "CellposeModel.eval"][-1]
    assert ev["z_axis"] == 0 and ev["stitch_threshold"] > 0
    pred = _imperfect(truth, _synthetic(1)[2], 1)
    missing = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()][0]
    pt = np.round(np.argwhere(truth == missing).mean(0)).astype(int)
    mask, box = b.segment_at(img, pt, pred)
    assert mask is not None and mask.shape == pred[box].shape
    # your own model file
    model = tmp_path / "mine"
    model.write_bytes(b"x")
    own = ai.CellposeBackend(pretrained=model)
    own.auto_segment(img)
    assert ("CellposeModel", {"pretrained_model": str(model), "model_type": None, "gpu": False}) in cellpose.CALLS
    # CellposeModel's own default is channels=None, so the adapter must pass [0, 0] itself
    assert [kw for n, kw in cellpose.CALLS if n == "CellposeModel.eval"][-1]["channels"] == [0, 0]
    assert not ai.CellposeBackend(pretrained=tmp_path / "gone").available()[0]


def test_micro_sam_adapter(fakes):
    import micro_sam
    b = ai.MicroSamBackend()
    assert b.available()[0] and not b.can_auto
    img, truth, c = _synthetic(0)
    pred = _imperfect(truth, c, 0)
    missing = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()]
    for i in missing:
        pt = tuple(np.round(np.argwhere(truth == i).mean(0)).astype(int))
        mask, box = b.segment_at(img, pt, pred)
        full = np.zeros(truth.shape, bool)
        full[box] = mask
        assert (full & (truth == i)).sum() / (full | (truth == i)).sum() > 0.8
        assert not (full & (pred > 0)).any()
    names = [n for n, _ in micro_sam.CALLS]
    assert names.count("get_sam_model") == 1 and names.count("precompute_image_embeddings") == 1   # cached
    assert [kw for n, kw in micro_sam.CALLS if n == "get_sam_model"][0] == {"model_type": "vit_t_lm", "device": "cpu"}
    with pytest.raises(NotImplementedError):
        b.auto_segment(img)


def test_yolo_adapter(fakes, tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    b = ai.YoloBackend(w)
    img, truth, _ = _synthetic(2)
    lab = b.auto_segment(img)
    assert lab.shape == truth.shape and metrics.match(truth, lab)["f1"] > 0.9
    with pytest.raises(ValueError):
        b.auto_segment(np.stack([img, img]))


def test_missing_packages_are_reported():
    for b in ai.backends_for(None)[1:]:
        ok, msg = b.available()
        if not ok:
            assert "install" in msg
            with pytest.raises(ai.BackendUnavailable):
                b.require()


def test_backends_find_project_models(tmp_path):
    p = make_demo_project(tmp_path / "p", 1)
    (p.root / "models" / "cellpose").mkdir(parents=True)
    (p.root / "models" / "cellpose" / "cells_20261010").write_bytes(b"x")
    (p.root / "models" / "cellpose" / "cells_20261010.json").write_text("{}")
    (p.root / "models" / "yolo").mkdir(parents=True)
    (p.root / "models" / "yolo" / "best.pt").write_bytes(b"x")
    p.set_meta("classic_params", {"sigma": 2.0})
    names = [b.name for b in ai.backends_for(p)]
    assert names[0].startswith("Classic") and sum("your model" in n for n in names) == 2
    assert ai.backends_for(p)[0].params["sigma"] == 2.0


# ------------------------------------------------------------------ batch (Phase 4)
def test_batch_with_classic(tmp_path):
    p = make_demo_project(tmp_path / "p", 3)
    stems = [i.stem for i in p.images()]
    out = list(batch.run_detector(p, "cells", stems, backend=ai.ClassicBackend()))
    assert all("skipped" in m for *_, m in out)                        # demo already has predictions
    out = list(batch.run_detector(p, "cells", stems, backend=ai.ClassicBackend(), overwrite=True))
    assert all("objects" in m for *_, m in out), out
    for s in stems:
        assert (p.predictions_dir / "cells" / f"{s}.tif").exists()
        assert batch.problem_value(p, s, "cells") >= 0
        pred, _ = p.prediction("cells", s, (384, 384))
        img, truth, _ = _synthetic(int(s[-2:]) - 1)
        assert metrics.match(truth, pred)["f1"] >= 0.95                  # classic is good on the demo
    # problem ranking of the faulty demo predictions -> highest scores
    q = make_demo_project(tmp_path / "q", 3)
    list(batch.rank_images(q, "cells", stems))
    assert all(batch.problem_value(q, s, "cells") > 0 for s in stems)


def test_batch_with_external_program(tmp_path):
    p = make_demo_project(tmp_path / "p", 2)
    script = tmp_path / "my detector.py"            # space in the path on purpose
    script.write_text(textwrap.dedent("""
        import sys, numpy as np, tifffile
        from scipy import ndimage as ndi
        args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        img = tifffile.imread(args["--input"]).astype(float)
        lab, n = ndi.label(ndi.gaussian_filter(img, 2) > img.mean() + img.std())
        tifffile.imwrite(args["--output"], lab.astype("uint16"))
    """))
    cmd = f'python "{script}" --input {{input}} --output {{output}}'
    out = list(batch.run_detector(p, "cells", ["demo_01", "demo_02"], command=cmd, overwrite=True))
    assert all("objects" in m for *_, m in out), out
    pred, _ = p.prediction("cells", "demo_01", (384, 384))
    assert pred.max() > 10
    bad = list(batch.run_detector(p, "cells", ["demo_01"], command='python -c "import sys; sys.exit(3)"',
                                  overwrite=True))
    assert "FAILED" in bad[0][3] and "code 3" in bad[0][3]
    silent = list(batch.run_detector(p, "cells", ["demo_01"], command='python -c "pass"', overwrite=True))
    assert "did not write" in silent[0][3]
    with pytest.raises(ValueError):
        list(batch.run_detector(p, "cells", ["demo_01"]))


def test_batch_cleanup_is_exact(tmp_path):
    p = make_demo_project(tmp_path / "p", 1)
    img = io.read_image(p.images()[0])
    import tifffile
    for name in ("img1", "img10"):
        tifffile.imwrite(str(p.images_dir / f"{name}.tif"), img)
    d = p.predictions_dir / "cells"
    d.mkdir(exist_ok=True)
    io.write_mask(d / "img10.tif", np.ones(img.shape, np.uint16))
    (d / "img1_mask.png").write_bytes(b"old")
    list(batch.run_detector(p, "cells", ["img1"], backend=ai.ClassicBackend(), overwrite=True))
    assert (d / "img10.tif").exists() and not (d / "img1_mask.png").exists() and (d / "img1.tif").exists()


def test_single_channel_selection():
    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[..., 1] = 90
    assert batch.single_channel(rgb, io.guess_layout(rgb)).max() == 30
    mc = np.stack([np.zeros((10, 10)), np.ones((10, 10))])
    assert batch.single_channel(mc, io.guess_layout(mc), 1).max() == 1
    assert batch.single_channel(mc, io.guess_layout(mc), 9).max() == 1      # out of range -> last


# ------------------------------------------------------------------ training (Phase 5)
def _dense_project(root, n=3):
    import tifffile

    from ani.project import Project
    p = Project(root)
    for i in range(n):
        im, lab = dense(seed=i)
        tifffile.imwrite(str(p.images_dir / f"d{i}.tif"), im)
        p.save_labels("cells", f"d{i}", lab, {}, lab.shape)
        p.set_status(f"d{i}", "done")
    return p


def test_evaluate_and_history(tmp_path):
    p = make_demo_project(tmp_path / "p", 2)
    for path in p.images():
        img, truth, _ = _synthetic(int(path.stem[-2:]) - 1)
        p.save_labels("cells", path.stem, truth, {}, truth.shape)
        p.set_status(path.stem, "done")
    r = train.evaluate(p, "cells")
    o = r["overall"]
    assert 0.5 < o["f1"] < 1.0 and o["fn"] >= 4 and o["fp"] >= 4      # the demo detector's planted errors
    r2 = train.evaluate(p, "cells", backend=ai.ClassicBackend())
    assert r2["overall"]["f1"] > o["f1"]
    assert len(p.meta["evaluations"]) == 2
    with pytest.raises(ValueError):
        train.evaluate(make_demo_project(tmp_path / "empty", 1), "cells")


def test_tune_classic_learns_from_corrections(tmp_path):
    p = _dense_project(tmp_path / "dense")
    steps = list(train.tune_classic(p, "cells", n_trials=24))
    kind, best, f1, old = steps[-1]
    assert kind == "done" and f1 >= old and f1 > 0.9, (f1, old)
    assert p.meta["classic_params"] == best
    after = train.evaluate(p, "cells", backend=ai.backends_for(p)[0], record=False)["overall"]["f1"]
    assert after > 0.9


def test_finetune_cellpose(fakes, tmp_path):
    import cellpose
    p = _dense_project(tmp_path / "p", n=3)
    path = train.finetune_cellpose(p, "cells", epochs=7)
    assert path.exists() and path.parent == p.root / "models" / "cellpose"
    assert not (p.root / "models" / "_cellpose_tmp").exists()
    call = [kw for n, kw in cellpose.CALLS if n == "train_seg"][-1]
    assert call["epochs"] == 7 and call["channels"] == [0, 0] and call["n_train"] == 3
    names = [b.name for b in ai.backends_for(p)]
    assert any(path.name in n for n in names)
    with pytest.raises(ValueError):
        train.finetune_cellpose(make_demo_project(tmp_path / "e", 1), "cells")


def test_colab_notebooks_are_valid(tmp_path):
    paths = train.write_notebooks(tmp_path, "cells")
    assert len(paths) == 2
    for path in paths:
        nb = json.loads(path.read_text())
        try:
            import nbformat
            nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
        except ImportError:
            assert nb["nbformat"] == 4
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                python = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("!"))
                ast.parse(python)          # every code cell is valid Python


def test_selftest_passes():
    from ani.__main__ import selftest
    assert selftest(check_ai=False) == 0


def test_external_python_in_frozen_app(monkeypatch, tmp_path):
    """In the installed .exe, 'python' must mean the system Python, not Ani.exe."""
    seen = {}

    def fake_run(args, **kw):
        seen["args"] = args
        raise RuntimeError("stop")
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError):
        batch.run_external("python det.py {input} {output}", tmp_path / "a.tif", tmp_path / "o.tif", "cells")
    assert seen["args"][0] == "python"
    monkeypatch.setattr(sys, "frozen", False)
    with pytest.raises(RuntimeError):
        batch.run_external("python det.py {input} {output}", tmp_path / "a.tif", tmp_path / "o.tif", "cells")
    assert seen["args"][0] == sys.executable


def _confluent():
    rng = np.random.default_rng(0)
    lab = np.zeros((300, 300), np.uint16)
    yy, xx = np.mgrid[:300, :300]
    k = 0
    for cy in range(20, 300, 40):
        for cx in range(20, 300, 40):
            k += 1
            lab[np.hypot(yy - cy, xx - cx) < 19] = k
    img = ndi.gaussian_filter((lab > 0) * rng.uniform(0.8, 1.0, k + 1)[lab] * 2000, 1) + 200
    return (img + rng.normal(0, 30, lab.shape)).astype(np.float32), lab


def test_polarity_on_confluent_cells(tmp_path):
    """Flat bright cells covering most of the image fool any intensity-only guess; labels settle it."""
    from ani.ai.base import polarity_from_labels
    img, lab = _confluent()
    assert polarity_from_labels(img, lab) == "bright"
    assert polarity_from_labels(img.max() - img, lab) == "dark"
    assert polarity_from_labels(img, np.zeros_like(lab)) is None
    assert metrics.match(lab, ai.ClassicBackend({"polarity": "bright"}).auto_segment(img))["f1"] > 0.95
    # tuning learns the polarity from the user's corrections
    import tifffile

    from ani.project import Project
    p = Project(tmp_path / "conf")
    tifffile.imwrite(str(p.images_dir / "c.tif"), img)
    p.save_labels("cells", "c", lab, {}, lab.shape)
    p.set_status("c", "done")
    kind, best, f1, old = list(train.tune_classic(p, "cells", n_trials=8))[-1]
    assert best["polarity"] == "bright" and f1 > 0.95


def test_channel_is_used_everywhere(tmp_path):
    """Cells only visible in channel 1: detect, rank, evaluate and tune must all look at channel 1."""
    import tifffile

    from ani.project import Project
    img, truth, _ = _synthetic(3)
    stack = np.stack([np.random.default_rng(0).normal(500, 20, img.shape).astype(np.uint16), img])
    p = Project(tmp_path / "mc")
    tifffile.imwrite(str(p.images_dir / "m.tif"), stack)
    p.save_labels("cells", "m", truth, {}, truth.shape)
    p.set_status("m", "done")
    good = train.evaluate(p, "cells", backend=ai.ClassicBackend(), channel=1, record=False)["overall"]["f1"]
    bad = train.evaluate(p, "cells", backend=ai.ClassicBackend(), channel=0, record=False)["overall"]["f1"]
    assert good > 0.95 > bad
    list(batch.run_detector(p, "cells", ["m"], backend=ai.ClassicBackend(), channel=1, overwrite=True))
    pred, _ = p.prediction("cells", "m", truth.shape)
    assert metrics.match(truth, pred)["f1"] > 0.95
    list(batch.rank_images(p, "cells", ["m"], channel=1))
    assert batch.problem_value(p, "m", "cells") == 0
    assert list(train.tune_classic(p, "cells", n_trials=4, channel=1))[-1][2] > 0.95


def test_download_models_command(fakes, capsys):
    from ani.__main__ import download_models
    assert download_models() == 0
    out = capsys.readouterr().out
    assert "Cellpose" in out and "micro-sam" in out and "YOLO" in out and "[FAIL]" not in out


def test_micro_sam_needs_mobile_sam(fakes, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "mobile_sam" else real(n, *a))
    ok, msg = ai.MicroSamBackend().available()
    assert not ok and "MobileSAM" in msg
