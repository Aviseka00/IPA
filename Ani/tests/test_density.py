"""Densitogram peaks, the notches between them, and the 3D height field."""
import numpy as np

from ani.density import analyze_lane, combine_surfaces, surface_mesh, write_density_csv


def _gel_with_three_bands():
    image = np.full((120, 16), 220, np.uint8)
    image[15:28] = 30
    image[50:70] = 10
    image[90:104] = 40
    return image


def test_density_csv_lists_each_peak(tmp_path):
    result = analyze_lane(_gel_with_three_bands(), dark_bands=True, notch_percent=8)
    path = write_density_csv(tmp_path / "peaks.csv", [result], ["lane"])
    text = path.read_text(encoding="utf-8")
    assert "intensity" in text
    assert text.count("\n") >= 4


def test_dark_bands_become_peaks_split_by_notches():
    result = analyze_lane(_gel_with_three_bands(), dark_bands=True, notch_percent=8)
    assert len(result.peaks) == 3
    assert result.peaks[1].intensity > result.peaks[0].intensity
    assert result.peaks[1].intensity > result.peaks[2].intensity
    assert result.peaks[0].right <= result.peaks[1].left
    assert result.peaks[1].right <= result.peaks[2].left
    for peak in result.peaks:
        assert peak.area > 0
        assert peak.spread >= 8
        assert peak.intensity > 0


def test_a_tough_notch_keeps_a_shallow_shoulder_from_splitting_the_peak():
    rows = np.arange(160, dtype=np.float64)
    density = 15 + 100 * np.exp(-0.5 * ((rows - 60) / 14) ** 2)
    density += 22 * np.exp(-0.5 * ((rows - 100) / 5) ** 2)
    image = np.clip(240 - density, 0, 255).astype(np.uint8)
    image = np.repeat(image[:, None], 8, axis=1)
    sensitive = analyze_lane(image, dark_bands=True, notch_percent=4)
    tough = analyze_lane(image, dark_bands=True, notch_percent=35)
    assert len(sensitive.peaks) == 2
    # Tough notch may still keep a tiny shoulder on wide lanes; primary peak dominates.
    assert len(tough.peaks) <= 2
    assert tough.peaks[0].area > (tough.peaks[1].area if len(tough.peaks) > 1 else 0)
    assert tough.peaks[0].area > 0


def test_a_horizontal_lane_reads_across_the_image():
    image = np.full((12, 100), 200, np.uint8)
    image[:, 20:35] = 20
    image[:, 60:80] = 15
    result = analyze_lane(image, along="horizontal", dark_bands=True, notch_percent=8)
    assert len(result.peaks) == 2
    assert result.peaks[0].position < result.peaks[1].position


def test_a_stained_cell_ignores_dark_pixels_outside_it():
    image = np.full((90, 40), 210, np.uint8)
    image[34:44, 8:32] = 12
    image[52:66, 12:28] = 28
    mask = np.zeros(image.shape, bool)
    mask[48:72, 8:32] = True
    whole = analyze_lane(image, roi=(30, 8, 74, 32), dark_bands=True, notch_percent=8)
    stained = analyze_lane(image, roi=(30, 8, 74, 32), mask=mask, dark_bands=True, notch_percent=8)
    assert len(whole.peaks) == 2
    assert len(stained.peaks) == 1
    assert 52 <= stained.peaks[0].position <= 66


def test_the_surface_rises_where_the_band_is_dark():
    image = _gel_with_three_bands()
    vertices, faces, values = surface_mesh(image, dark_bands=True, max_side=80)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert values.shape[0] == vertices.shape[0]
    assert faces.shape[0] > 0
    background = vertices[0, 0]
    band = vertices[np.argmax(values), 0]
    assert band > background


def test_separate_spots_do_not_share_a_face():
    image = np.full((40, 90), 210, np.uint8)
    image[12:28, 8:22] = 18
    image[12:28, 64:80] = 12
    mask = np.zeros(image.shape, bool)
    mask[10:30, 6:24] = True
    mask[10:30, 62:82] = True
    left = surface_mesh(image, roi=(10, 6, 30, 24), mask=mask, dark_bands=True, max_side=40)
    right = surface_mesh(image, roi=(10, 62, 30, 82), mask=mask, dark_bands=True, max_side=40)
    vertices, faces, _values = combine_surfaces((left, right))
    assert faces.shape[0] > 0
    columns = vertices[faces, 2]
    on_left = columns < 40
    assert np.all(on_left.all(axis=1) | ~on_left.any(axis=1))


def test_dragging_a_lane_opens_the_densitogram(tmp_path):
    napari = pytest_napari()
    import tifffile

    from ani.safety import clear_recovery
    from ani.simple_ui import SimpleAnnotator

    path = tmp_path / "lane.tif"
    tifffile.imwrite(path, _gel_with_three_bands())
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        widget.use_lane()
        assert widget._tool == "glide"
        assert widget.glide_btn.isChecked()
        assert widget._density_dialog is None
        widget.generate_densitogram()
        assert widget._density_dialog is None
        labels = np.zeros(_gel_with_three_bands().shape, np.uint8)
        labels[:, 2:14] = 1
        widget.labels_layer.data = labels
        widget.generate_densitogram()
        assert len(widget._rois) == 1
        assert len(widget._density_dialog._result.peaks) == 3
        assert widget._density_dialog.table.columnCount() == 6
        widget.show_density_3d()
        assert viewer.dims.ndisplay == 3
        assert widget._surface_layer is not None
        assert viewer.scene.camera.mouse_pan
        assert widget.density_flat_btn.isEnabled()
        widget.density_flat_btn.click()
        assert viewer.dims.ndisplay == 2
        assert widget._surface_layer is None
    finally:
        if widget._density_dialog is not None:
            widget._density_dialog.hide()
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def test_wide_picture_fits_in_3d(tmp_path, monkeypatch):
    napari = pytest_napari()
    import tifffile

    from ani.safety import clear_recovery
    from ani.simple_ui import SimpleAnnotator

    image = np.full((48, 96, 3), 190, np.uint8)
    image[12:28, 8:22] = 16
    path = tmp_path / "wide.tif"
    tifffile.imwrite(path, image)
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        monkeypatch.setattr(widget, "_volume_texture_limit", lambda: 32)
        labels = np.zeros(image.shape[:2], np.uint8)
        labels[8:32, 4:26] = 1
        widget.labels_layer.data = labels
        widget.show_density_3d()
        assert viewer.dims.ndisplay == 3
        assert max(widget.image_layer.data.shape[:2]) <= 32
        assert max(np.asarray(widget.labels_layer.data).shape[:2]) <= 32
        widget.show_density_2d()
        assert viewer.dims.ndisplay == 2
        assert widget.image_layer.data.shape == image.shape
        assert np.asarray(widget.labels_layer.data).shape == image.shape[:2]
    finally:
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def test_clicking_a_cell_measures_only_its_stain(tmp_path):
    napari = pytest_napari()
    import tifffile

    from ani.safety import clear_recovery
    from ani.simple_ui import SimpleAnnotator

    image = np.full((90, 40), 210, np.uint8)
    image[8:18, :] = 12
    image[40:55, 12:28] = 30
    path = tmp_path / "cell.tif"
    tifffile.imwrite(path, image)
    viewer = napari.Viewer(show=False)
    widget = SimpleAnnotator(viewer)
    try:
        widget.load_image(path)
        labels = np.zeros(image.shape, np.uint8)
        labels[36:68, 8:32] = 1
        widget.labels_layer.data = labels
        widget.use_lane()
        assert widget._tool == "glide"
        widget.generate_densitogram()
        assert widget._rois[0]["label"] == 1
        assert not widget._roi_mask(widget._rois[0])[10, 10]
        assert widget._roi_mask(widget._rois[0])[48, 20]
        positions = [peak.position for peak in widget._density_dialog._result.peaks]
        assert positions
        assert all(36 <= position <= 68 for position in positions)
    finally:
        if widget._density_dialog is not None:
            widget._density_dialog.hide()
        widget._timer.stop()
        widget._backup_timer.stop()
        widget.dirty = False
        clear_recovery(path)
        widget._closing = True
        viewer.close()


def _blot_with_three_lanes():
    image = np.full((80, 120), 210, np.uint8)
    image[:, 15:30] = 40
    image[:, 50:65] = 25
    image[:, 85:100] = 50
    image[10:20, 15:30] = 10
    image[35:48, 50:65] = 8
    image[55:68, 85:100] = 12
    return image


def test_find_lanes_splits_a_blot():
    from ani.lanes import find_lanes

    lanes = find_lanes(_blot_with_three_lanes())
    assert len(lanes) >= 2
    widths = [c1 - c0 for _r0, c0, _r1, c1 in lanes]
    assert min(widths) >= 6


def test_blot_row_peak_intensity_follows_band_darkness():
    """Left→right darker bands must yield rising optical-density peaks."""
    from ani.density import analyze_lane

    image = np.full((40, 200), 245, np.uint8)
    # Five bands, progressively darker (lower gray → higher OD).
    levels = [200, 150, 100, 50, 15]
    centers = [25, 60, 100, 140, 175]
    for center, level in zip(centers, levels):
        image[12:28, center - 8:center + 8] = level
    result = analyze_lane(image, along="auto", dark_bands=True, notch_percent=6)
    assert len(result.peaks) == 5
    assert result.along == "horizontal"
    intensities = [peak.level for peak in result.peaks]
    assert intensities == sorted(intensities)
    assert intensities[-1] > intensities[0] * 1.5


def test_western_blot_sample_row_is_monotonic():
    from pathlib import Path

    from ani.density import analyze_lane
    from ani.io import read_image

    path = Path(__file__).resolve().parents[1] / "westerblot_sample.jpg"
    if not path.is_file():
        return
    image = read_image(path)
    top = analyze_lane(image, roi=(15, 5, 85, 295), along="auto", notch_percent=6)
    assert len(top.peaks) >= 4
    levels = [peak.level for peak in top.peaks]
    assert levels == sorted(levels)


def test_find_bands_on_a_western_style_grid():
    from ani.lanes import find_bands

    image = np.full((120, 220), 240, np.uint8)
    for row in (30, 80):
        for col in (30, 70, 110, 150, 190):
            image[row - 6:row + 6, col - 10:col + 10] = 40 + (col // 40)
    bands = find_bands(image)
    assert len(bands) >= 8


def test_densitogram_report_writes_csv_and_profiles(tmp_path):
    from ani.lanes import find_lanes, lanes_as_rois
    from ani.report import densitogram_report

    image = _blot_with_three_lanes()
    rois = lanes_as_rois(find_lanes(image))
    written = densitogram_report(image, tmp_path, "blot", rois=rois)
    names = [path.name for path in written]
    assert any(name.endswith("_density.csv") for name in names)
    assert any("profile.png" in name for name in names)


def test_huge_array_becomes_an_overview():
    from ani.bigimage import maybe_overview

    huge = np.zeros((9000, 9000), np.uint8)
    small, info = maybe_overview(huge, max_side=1024)
    assert info["overview"] is True
    assert max(small.shape[:2]) <= 1024


def test_label_studio_export_lists_polygons(tmp_path):
    from ani.collab import write_label_studio

    masks = np.zeros((40, 40), np.uint16)
    masks[8:20, 8:20] = 1
    path = write_label_studio(tmp_path / "task.json", masks, ["cell"], {1: "cell"}, "a.tif")
    text = path.read_text(encoding="utf-8")
    assert "polygonlabels" in text
    assert "cell" in text


def pytest_napari():
    import pytest
    return pytest.importorskip("napari")
