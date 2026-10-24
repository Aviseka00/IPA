"""Real-library tests (Cellpose downloads its model). Run with:  ANI_INTEGRATION=1 pytest -m integration

These run in CI (see .github/workflows/ci.yml) and on any PC with Cellpose installed and internet access.
"""
import os

import numpy as np
import pytest

from ani import ai, metrics, train
from ani.demo import _synthetic

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("ANI_INTEGRATION") != "1", reason="set ANI_INTEGRATION=1")]


@pytest.fixture(autouse=True)
def need_cellpose():
    pytest.importorskip("cellpose")


def test_real_cellpose_segments_demo_cells():
    img, truth, _ = _synthetic(0)
    b = ai.CellposeBackend("cyto3")
    lab = b.auto_segment(img)
    assert lab.shape == truth.shape
    assert metrics.match(truth, lab)["f1"] > 0.8


def test_real_cellpose_click():
    from ani import tools
    from ani.demo import _imperfect
    img, truth, c = _synthetic(0)
    pred = _imperfect(truth, c, 0)
    missing = [i for i in tools.label_ids(truth) if not (pred[truth == i] > 0).any()][0]
    pt = np.round(np.argwhere(truth == missing).mean(0)).astype(int)
    mask, box = ai.CellposeBackend("cyto3").segment_at(img, pt, pred)
    assert mask is not None
    full = np.zeros(truth.shape, bool)
    full[box] = mask
    assert (full & (truth == missing)).sum() / (full | (truth == missing)).sum() > 0.7


def test_real_cellpose_finetune(tmp_path):
    from ani.demo import make_demo_project
    p = make_demo_project(tmp_path / "p", n_images=3)
    for path in p.images():
        img, truth, _ = _synthetic(int(path.stem[-2:]) - 1)
        p.save_labels("cells", path.stem, truth, {}, truth.shape)
        p.set_status(path.stem, "done")
    model = train.finetune_cellpose(p, "cells", epochs=3)
    assert model.exists()
    own = [b for b in ai.backends_for(p) if model.name in b.name][0]
    lab = own.auto_segment(_synthetic(5)[0])
    assert lab.max() > 0
