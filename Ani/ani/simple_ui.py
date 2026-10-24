"""A single-screen annotator: open an image, draw cell outlines, right-click to set the class, save."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEvent, QObject, QSize, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QColor, QCursor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import io, tools
from .features import export_training_set
from .review import brush_indices, fill_label_holes, folder_progress, review_message, review_outlines
from .safety import (
    MAX_FEATURE_PIXELS, clear_recovery, feature_job_error, load_recovery, log_error, remember_image,
    recent_images, save_recovery,
)
from .seg import SAVE_FILTER, SAVE_KINDS, kind_from_filter, mask_from_glide, prepare_for_display, read_seg, write_segmentation
from .segment import run_click, run_segment, segment_choices

PALETTE = ["#5B8DEF", "#E3B341", "#3DDC97", "#FF7B72", "#C792EA", "#FF9B4A", "#39C5CF", "#F778BA"]
CLASS_COLORS = {"infected": "#FF5A4E", "duplicate": "#3DDC97"}


def _fit_plane(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize the first two axes. Integer arrays keep their labels."""
    from scipy import ndimage as ndi

    array = np.asarray(array)
    factors = [height / array.shape[0], width / array.shape[1]]
    if array.ndim == 3:
        factors.append(1.0)
    order = 0 if np.issubdtype(array.dtype, np.integer) else 1
    small = ndi.zoom(array, factors, order=order)
    small = small[:height, :width]
    if array.ndim == 3:
        small = small[:, :, : array.shape[2]]
    if np.issubdtype(array.dtype, np.integer):
        small = np.rint(small).astype(array.dtype, copy=False)
    return np.ascontiguousarray(small)
IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.webp *.gif)"


def _swatch(color: str) -> QIcon:
    pix = QPixmap(14, 14)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 12, 12)
    painter.end()
    return QIcon(pix)


def _block_label_shortcut(layer):
    """Hold a napari label key so it does not switch the paint mode."""
    return None


def _hex(color) -> str:
    try:
        return color.as_hex()
    except Exception:
        return str(color)


class _FeatureJob(QThread):
    """Measure cells off the drawing thread so the window stays responsive."""

    progressed = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, payload: dict):
        super().__init__()
        self.payload = payload

    def run(self):
        try:
            written = export_training_set(**self.payload, progress=self.progressed.emit)
            self.succeeded.emit(str(written))
        except Exception as exc:
            log_error("feature extraction", exc)
            self.failed.emit(str(exc))


class _SegmentJob(QThread):
    """Label objects off the drawing thread. A point set means one click, not the whole image."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, image, key: str, point=None, labels=None):
        super().__init__()
        self.image = image
        self.key = key
        self.point = point
        self.labels = labels

    def run(self):
        try:
            if self.point is None and self.key == "spots":
                from .spots import find_infected

                labels, classes, meta = find_infected(self.image)
                self.succeeded.emit({"labels": labels, "classes": classes, "meta": meta})
            elif self.point is None:
                self.succeeded.emit(run_segment(self.image, self.key))
            else:
                self.succeeded.emit(run_click(self.image, self.labels, self.point, self.key))
        except Exception as exc:
            log_error("segment", exc)
            self.failed.emit(str(exc))


class _PrefetchJob(QThread):
    """Decode the next folder image in the background."""

    succeeded = Signal(str, object)
    failed = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)

    def run(self):
        try:
            from .bigimage import read_for_display
            from .cache import IMAGE_CACHE

            if IMAGE_CACHE.get(self.path) is not None:
                return
            image, _info = read_for_display(self.path)
            IMAGE_CACHE.put(self.path, image)
            self.succeeded.emit(str(self.path), image)
        except Exception as exc:
            log_error("prefetch", exc)
            self.failed.emit(str(exc))


class _BatchReportJob(QThread):
    """Build densitogram reports for a whole folder off the UI thread."""

    progressed = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path, out_dir: Path, notch: float, dark: bool):
        super().__init__()
        self.folder = Path(folder)
        self.out_dir = Path(out_dir)
        self.notch = float(notch)
        self.dark = bool(dark)

    def run(self):
        try:
            from .report import batch_densitogram_reports

            written = batch_densitogram_reports(
                self.folder,
                out_dir=self.out_dir,
                dark_bands=self.dark,
                notch_percent=self.notch,
                progress=lambda done, total, name: self.progressed.emit(done, total, name),
            )
            self.succeeded.emit(written)
        except Exception as exc:
            log_error("batch report", exc)
            self.failed.emit(str(exc))


class _GrayJob(QThread):
    """Precompute brightness so Label now starts instantly."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        try:
            from .segment import to_gray

            self.succeeded.emit(np.ascontiguousarray(to_gray(self.image), dtype=np.float32))
        except Exception as exc:
            log_error("gray prefetch", exc)
            self.failed.emit(str(exc))


class _CloseFilter(QObject):
    def __init__(self, widget: SimpleAnnotator):
        super().__init__()
        self.widget = widget

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if event.type() == QEvent.Close and not self.widget._closing:
            if not self.widget.confirm_discard():
                event.ignore()
                return True
            self.widget._closing = True
        elif event.type() == QEvent.Close:
            self.widget._backup()
        return False


class SimpleAnnotator(QWidget):
    """Draw cell boundaries and give each cell a class."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.image_path: Path | None = None
        self._display_image = None
        self.display_info: dict = {}
        self.labels_layer = None
        self.class_names = ["cell"]
        self.classes: dict[int, str] = {}
        self._known: set[int] = set()
        self._snapshot: tuple[np.ndarray, dict] | None = None
        self.undo_stack: list[tuple[np.ndarray, dict, int]] = []
        self.redo_stack: list[tuple[np.ndarray, dict, int]] = []
        self._rev = 0
        self._saved_rev = 0
        self.dirty = False
        self._internal = False
        self._in_flush = False
        self._closing = False
        self._colors: dict[str, str] = {}
        self._tool = "glide"
        self._glide_layer = None
        self._glide_pts: list[np.ndarray] = []
        self._glide_cancel = False
        self._cursor = None
        self._feature_job = None
        self._restored = False
        self._poly_layer = None
        self._poly_pts: list[np.ndarray] = []
        self._poly_click: tuple[float, np.ndarray] | None = None
        self._segment_job = None
        self._segment_token = 0
        self._stroking = False
        self._stroke_data = None
        self._brush_last = None
        self._paint_drawn = 0.0
        self._review: dict = {}
        self.image_layer = None
        self._rois = []
        self._lane_roi = None
        self._density_mask = None
        self._density_kind = None
        self._lane_layer = None
        self._surface_layer = None
        self._density_dialog = None
        self._density_along = "auto"
        self._density_dark = True
        self._viewing_3d = False
        self._volume_backup = None
        self._fitting_volume = False
        self._density_notch = 5.0
        self._spot_pairs: dict[int, int] = {}
        self._hidden_classes: set[str] = set()
        self._folder_images: list[Path] = []
        self._segment_draft = False
        self._light_ui = False
        self._prefetch_job = None
        self._batch_job = None
        self._gray_cache = None
        self._gray_job = None

        self._build()
        self._apply_style()
        self._bind_keys()
        self._guard_display_button()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.flush)
        self._backup_timer = QTimer(self)
        self._backup_timer.setInterval(20000)
        self._backup_timer.timeout.connect(self._backup)
        self._backup_timer.start()
        self._review_timer = QTimer(self)
        self._review_timer.setSingleShot(True)
        self._review_timer.setInterval(250)
        self._review_timer.timeout.connect(self._run_review)
        self._refresh_buttons()
        QTimer.singleShot(0, self._quiet_viewer)
        try:
            self._close_filter = _CloseFilter(self)
            viewer.window._qt_window.installEventFilter(self._close_filter)
        except Exception:
            pass

    def sizeHint(self):  # noqa: N802 (Qt naming)
        return QSize(348, 810)

    @property
    def stem(self) -> str | None:
        return None if self.image_path is None else self.image_path.stem

    # ------------------------------------------------------------------- appearance
    def _build(self):
        self.setObjectName("aniHost")
        self.setMinimumWidth(332)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body = QWidget()
        body.setObjectName("aniSimple")
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self._body = body

        root = QVBoxLayout(body)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(4)

        title = QLabel("Ani")
        title.setObjectName("title")
        root.addWidget(title)
        self.file_label = QLabel("No image open")
        self.file_label.setObjectName("file")
        self.file_label.setWordWrap(True)
        root.addWidget(self.file_label)

        self.open_btn = self._button("Open image", self.open_image, "primary")
        self.open_btn.setToolTip("Open a picture (Ctrl+O). You can also drop a file here.")
        root.addWidget(self.open_btn)
        root.addLayout(self._row(
            ("Open folder", self.open_folder, "Open every image in a folder and step through them"),
            ("Recent", self.show_recent, "Images opened on this computer"),
            names=("folder_btn", "recent_btn"),
        ))
        root.addLayout(self._row(
            ("Previous", self.previous_image, "Previous image in this folder"),
            ("Next", self.next_image, "Next image in this folder"),
            names=("prev_btn", "next_btn"),
        ))
        self.folder_list = QListWidget()
        self.folder_list.setFixedHeight(64)
        self.folder_list.setSpacing(0)
        self.folder_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.folder_list.setToolTip("Images in this folder. Click one to open it.")
        self.folder_list.itemClicked.connect(self._open_folder_item)
        root.addWidget(self.folder_list)

        root.addWidget(self._kicker("Class"))
        self.class_list = QListWidget()
        self.class_list.setFixedHeight(52)
        self.class_list.setSpacing(1)
        self.class_list.setIconSize(QSize(12, 12))
        self.class_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.class_list.currentItemChanged.connect(lambda *_: self._recolor())
        root.addWidget(self.class_list)
        self.add_btn = self._button("Add class", lambda: self.add_class(None))
        self.add_btn.setToolTip("Add a class, such as healthy, dead, or infected")
        root.addWidget(self.add_btn)
        self.show_infected_btn = self._button("Infected", self._toggle_infected_visible)
        self.show_infected_btn.setCheckable(True)
        self.show_infected_btn.setChecked(True)
        self.show_infected_btn.setToolTip("Show or hide infected cells")
        self.show_duplicate_btn = self._button("Duplicates", self._toggle_duplicate_visible)
        self.show_duplicate_btn.setCheckable(True)
        self.show_duplicate_btn.setChecked(True)
        self.show_duplicate_btn.setToolTip("Show or hide duplicate copies")
        self.pair_btn = self._button("Find pair", self.find_selected_pair)
        self.pair_btn.setToolTip("Jump between an infected cell and its duplicate copy")
        root.addLayout(self._row_widgets(self.show_infected_btn, self.show_duplicate_btn, self.pair_btn))
        self._refresh_class_list("cell")

        root.addWidget(self._kicker("Tools"))
        self.glide_btn = self._button("Glide", self.use_glide)
        self.points_btn = self._button("Points", self.use_points)
        self.pan_btn = self._button("Pan", self.use_pan)
        self.brush_btn = self._button("Brush", self.use_brush)
        self.erase_btn = self._button("Erase", self.use_erase)
        self.click_btn = self._button("Click", self.use_click)
        for button in (self.glide_btn, self.points_btn, self.pan_btn, self.brush_btn, self.erase_btn, self.click_btn):
            button.setCheckable(True)
        self.glide_btn.setChecked(True)
        self.glide_btn.setToolTip("Drag around a spot to outline it. Pan moves the image. Ctrl+click places corners. (G)")
        self.points_btn.setToolTip("Click corners, then double-click to close. Drag moves the image. (P)")
        self.pan_btn.setToolTip("Drag moves the image. The wheel zooms. (Z)")
        self.brush_btn.setToolTip("Drag to extend an object or start a new one. (B)")
        self.erase_btn.setToolTip("Drag to remove outline pixels. (E)")
        self.click_btn.setToolTip("Click one object to outline only that object. (C)")
        group = QButtonGroup(self)
        group.setExclusive(True)
        self._tool_group = group
        tools = QGridLayout()
        tools.setSpacing(4)
        for index, button in enumerate(
            (self.glide_btn, self.points_btn, self.pan_btn, self.brush_btn, self.erase_btn, self.click_btn)
        ):
            group.addButton(button)
            tools.addWidget(button, index // 3, index % 3)
        for column in range(3):
            tools.setColumnStretch(column, 1)
        root.addLayout(tools)

        self.brush_label = QLabel("Brush 8 px")
        self.brush_label.setObjectName("hint")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(2, 40)
        self.brush_slider.setValue(8)
        self.brush_slider.valueChanged.connect(self._set_brush_size)
        root.addLayout(self._slider_row(self.brush_label, self.brush_slider))
        self.opacity_label = QLabel("Fill 55%")
        self.opacity_label.setObjectName("hint")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(15, 100)
        self.opacity_slider.setValue(55)
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        root.addLayout(self._slider_row(self.opacity_label, self.opacity_slider))

        self.fill_btn = self._button("Fill", self.fill_holes)
        self.check_btn = self._button("Check", self.check_outlines)
        self.fit_btn = self._button("Fit", self.fit_view)
        self.contrast_btn = self._button("Contrast", self.auto_contrast)
        self.fill_btn.setToolTip("Close gaps trapped inside a cell (F)")
        self.check_btn.setToolTip("Look for holes, edge cuts, tiny fragments and missing classes")
        self.fit_btn.setToolTip("Show the whole image")
        self.contrast_btn.setToolTip("Set brightness from the image")
        fixes = QGridLayout()
        fixes.setSpacing(4)
        for index, button in enumerate((self.fill_btn, self.check_btn, self.fit_btn, self.contrast_btn)):
            fixes.addWidget(button, 0, index)
            fixes.setColumnStretch(index, 1)
        root.addLayout(fixes)

        self.draw_hint = QLabel(
            "Drag with Glide around a spot. Pan moves the image."
        )
        self.draw_hint.setObjectName("hint")
        self.draw_hint.setWordWrap(True)
        root.addWidget(self.draw_hint)

        root.addWidget(self._kicker("Segment"))
        self.segment_choice = QComboBox()
        for key, label in segment_choices():
            self.segment_choice.addItem(label, key)
        self.segment_choice.setToolTip("Infected spots finds the small dark cells, and marks a repeated copy as duplicate")
        self.segment_choice.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.segment_choice.setMinimumContentsLength(10)
        self.segment_btn = self._button("Segment", self.segment_image, "primary")
        self.segment_btn.setToolTip("Label the whole image, then correct with the brush")
        self.label_now_btn = self._button("Label now", self.label_infected_now)
        self.label_now_btn.setToolTip("Find infected cells and mark duplicate copies in one step")
        segment = QHBoxLayout()
        segment.setSpacing(4)
        segment.addWidget(self.segment_choice, 1)
        segment.addWidget(self.segment_btn, 0)
        root.addLayout(segment)
        root.addWidget(self.label_now_btn)
        self.accept_draft_btn = self._button("Keep draft", self.accept_segment_draft, "primary")
        self.accept_draft_btn.setToolTip("Keep the automatic outlines. You can still edit them with the brush.")
        self.reject_draft_btn = self._button("Reject draft", self.reject_segment_draft)
        self.reject_draft_btn.setToolTip("Remove the automatic outlines (same as Undo)")
        self.accept_draft_btn.hide()
        self.reject_draft_btn.hide()
        root.addLayout(self._row_widgets(self.accept_draft_btn, self.reject_draft_btn))

        self.summary = QLabel("No cells yet")
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        root.addLayout(self._row(
            ("Undo", self.undo, "Ctrl+Z"),
            ("Redo", self.redo, "Ctrl+Shift+Z"),
            names=("undo_btn", "redo_btn"),
        ))

        self.save_btn = self._button("Save annotations", self.save, "primary")
        self.save_btn.setToolTip("seg.npy, TIFF, PNG, COCO, YOLO, or all of them (Ctrl+S)")
        root.addWidget(self.save_btn)
        self.package_btn = self._button("Save package", self.save_package)
        self.package_btn.setToolTip("Save outlines and a densitogram CSV beside the image")
        root.addWidget(self.package_btn)

        root.addWidget(self._kicker("Features"))
        self.well_edit = QLineEdit()
        self.slide_edit = QLineEdit()
        self.mag_edit = QLineEdit()
        self.pixel_edit = QLineEdit()
        self.well_edit.setPlaceholderText("Well")
        self.slide_edit.setPlaceholderText("Slide")
        self.mag_edit.setPlaceholderText("Magnification")
        self.pixel_edit.setPlaceholderText("µm / pixel")
        self.pixel_edit.setToolTip("Calibration. Sizes in the feature table are converted when this is set")
        fields = QGridLayout()
        fields.setSpacing(4)
        for index, box in enumerate((self.well_edit, self.slide_edit, self.mag_edit, self.pixel_edit)):
            fields.addWidget(box, index // 2, index % 2)
        fields.setColumnStretch(0, 1)
        fields.setColumnStretch(1, 1)
        root.addLayout(fields)
        self.feature_btn = self._button("Extract features", self.extract_features)
        self.feature_btn.setToolTip(
            "Save a table, crops, masks, 128×128 images and a check picture. Uses the outlines you kept."
        )
        root.addWidget(self.feature_btn)

        root.addWidget(self._kicker("Density"))
        root.addLayout(self._row(
            ("Mark ROI", self.use_lane, "Then draw each spot with Glide, Points, or the brush, the same way you annotate (D)"),
            ("Generate", self.generate_densitogram, "Graph and table for every area you outlined: intensity, peak height, and area"),
            names=("density_btn", "generate_btn"),
        ))
        root.addLayout(self._row(
            ("Find lanes", self.find_gel_lanes, "Auto-detect vertical gel lanes like ImageJ, then open the densitogram"),
            ("Find bands", self.find_gel_bands, "Auto-detect individual blot/gel bands (any layout), then open the densitogram"),
            ("Report", self.one_click_report, "Save densitogram CSV + profile PNGs beside the image"),
            names=("lanes_btn", "report_btn"),
        ))
        root.addLayout(self._row(
            ("Batch report", self.batch_folder_report, "Densitogram CSV + profiles for every image in this folder"),
            names=("batch_report_btn",),
        ))
        root.addLayout(self._row(
            ("3D", self.show_density_3d, "Turn the outlined spots and see their intensity. Drag rotates"),
            ("Normal", self.show_density_2d, "Put the picture back in its usual flat position, facing you"),
            names=("density_3d_btn", "density_flat_btn"),
        ))

        root.addWidget(self._kicker("Share"))
        root.addLayout(self._row(
            ("Share package", self.share_package_folder, "Copy image + masks + COCO into a shared folder for teammates"),
            ("Label Studio", self.export_label_studio, "Export polygons for Label Studio / collaborator review"),
            names=("share_btn", "label_studio_btn"),
        ))

        root.addLayout(self._row(
            ("Help", self.show_help, "Short steps for labeling and densitograms, plus the log folder"),
            ("Light / Dark", self.toggle_panel_theme, "Switch the side panel between light and dark for long sessions"),
            names=("help_btn", "theme_btn"),
        ))
        self.status = QLabel("Open an image to start.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        root.addStretch(1)

    def _kicker(self, text: str) -> QLabel:
        lab = QLabel(text.upper())
        lab.setObjectName("kicker")
        return lab

    def _button(self, text, fn, kind="") -> QPushButton:
        button = QPushButton(text)
        if kind:
            button.setObjectName(kind)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setMinimumHeight(28)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda *_: fn())
        return button

    def _row(self, *items, names=()) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        for index, (text, fn, tip) in enumerate(items):
            button = self._button(text, fn)
            button.setToolTip(tip)
            if index < len(names):
                setattr(self, names[index], button)
            row.addWidget(button, 1)
        return row

    def _row_widgets(self, *widgets) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        for widget in widgets:
            row.addWidget(widget, 1)
        return row

    def _slider_row(self, caption: QLabel, slider: QSlider) -> QHBoxLayout:
        caption.setFixedWidth(78)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(caption, 0)
        row.addWidget(slider, 1)
        return row

    def toggle_panel_theme(self):
        self._light_ui = not self._light_ui
        self._apply_style()
        self._set_status("Light panel on." if self._light_ui else "Dark panel on.")

    def _apply_style(self):
        if self._light_ui:
            accent, bg, surface = "#3b5bdb", "#f4f6f8", "#ffffff"
            text, border, muted = "#1b1f2a", "#c5cad6", "#5c6578"
            checked = "#dbe4ff"
            primary_text = "#ffffff"
        else:
            accent, bg, surface = "#6f8cff", "#171a21", "#232833"
            text, border, muted = "#f4f6f8", "#3c4454", "#a7b0be"
            checked = "#31406f"
            primary_text = "#12141a"
        self._colors = {
            "accent": accent, "bg": bg, "surface": surface, "text": text, "muted": muted,
            "primary_text": primary_text,
        }
        family = '"Segoe UI Variable Text", "Segoe UI", sans-serif'
        self.setStyleSheet(f"""
            #aniHost, #aniSimple, QScrollArea, QScrollArea > QWidget > QWidget {{
                background: {bg}; color: {text}; border: none;
            }}
            #aniSimple {{
                font-family: {family}; font-size: 13px;
            }}
            #aniSimple QLabel {{ color: {text}; background: transparent; }}
            #aniSimple QLabel#title {{
                font-size: 18px; font-weight: 600; letter-spacing: 0.2px;
            }}
            #aniSimple QLabel#kicker {{
                color: {muted}; font-size: 11px; font-weight: 600; letter-spacing: 0.6px;
                padding-top: 2px;
            }}
            #aniSimple QLabel#hint, #aniSimple QLabel#file, #aniSimple QLabel#status {{
                color: {muted}; font-size: 12px;
            }}
            #aniSimple QLabel#summary {{ font-size: 13px; font-weight: 600; }}
            #aniSimple QPushButton {{
                background: {surface}; color: {text}; border: 1px solid {border};
                border-radius: 7px; padding: 2px 8px; min-height: 26px;
                font-size: 13px; font-weight: 500;
            }}
            #aniSimple QPushButton:hover {{ border-color: {accent}; }}
            #aniSimple QPushButton:checked {{
                background: {checked}; color: {text}; border: 1px solid {accent}; font-weight: 600;
            }}
            #aniSimple QPushButton#primary {{
                background: {accent}; color: {primary_text}; border: none; font-weight: 600;
            }}
            #aniSimple QPushButton#primary:hover {{ background: #8aa2ff; }}
            #aniSimple QPushButton:disabled {{ color: {muted}; background: {bg}; border-color: {border}; }}
            #aniSimple QListWidget {{
                background: {surface}; color: {text}; border: 1px solid {border};
                border-radius: 7px; padding: 2px; outline: none; font-size: 13px;
            }}
            #aniSimple QListWidget::item {{ padding: 3px 8px; border-radius: 5px; }}
            #aniSimple QListWidget::item:selected {{ background: {accent}; color: {primary_text}; font-weight: 600; }}
            #aniSimple QLineEdit, #aniSimple QComboBox {{
                background: {surface}; color: {text}; border: 1px solid {border};
                border-radius: 7px; padding: 2px 8px; min-height: 26px; font-size: 13px;
            }}
            #aniSimple QComboBox QAbstractItemView {{
                background: {surface}; color: {text}; selection-background-color: {accent};
                selection-color: {primary_text}; font-size: 13px; outline: none;
            }}
            #aniSimple QSlider::groove:horizontal {{
                height: 4px; background: {border}; border-radius: 2px;
            }}
            #aniSimple QSlider::handle:horizontal {{
                background: {accent}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }}
            QScrollBar:vertical {{
                background: {bg}; width: 10px; margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {border}; border-radius: 4px; min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    def _bind_keys(self):
        self.viewer.bind_key("Control-O", lambda v: self.open_image(), overwrite=True)
        self.viewer.bind_key("Control-S", lambda v: self.save(), overwrite=True)
        self.viewer.bind_key("Control-Z", lambda v: self.undo(), overwrite=True)
        self.viewer.bind_key("Control-Shift-Z", lambda v: self.redo(), overwrite=True)
        self.viewer.bind_key("Escape", lambda v: self._cancel_drawing(), overwrite=True)
        self._release_napari_keys()

    def _release_napari_keys(self):
        """Keep number keys and drawing letters for Ani instead of napari's paint modes."""
        from napari.utils.action_manager import action_manager
        names = (
            "napari:activate_labels_erase_mode", "napari:activate_labels_paint_mode",
            "napari:activate_labels_polygon_mode", "napari:activate_labels_fill_mode",
            "napari:activate_labels_picker_mode", "napari:activate_labels_pan_zoom_mode",
            "napari:activate_labels_transform_mode", "napari:new_label",
            "napari:swap_selected_and_background_labels", "napari:decrease_label_id",
            "napari:increase_label_id", "napari:decrease_brush_size", "napari:increase_brush_size",
            "napari:toggle_preserve_labels", "napari:reset_polygon",
        )
        for name in names:
            if name in getattr(action_manager, "_actions", {}) or name in getattr(action_manager, "_shortcuts", {}):
                try:
                    action_manager.unbind_shortcut(name)
                except Exception:
                    pass

    def _arm_canvas_keys(self, layer):
        """Keys on the image itself. These win over napari's label shortcuts."""
        layer.bind_key("Control-Z", lambda lyr: self.undo(), overwrite=True)
        layer.bind_key("Control-Shift-Z", lambda lyr: self.redo(), overwrite=True)
        layer.bind_key("Control-S", lambda lyr: self.save(), overwrite=True)
        layer.bind_key("Control-O", lambda lyr: self.open_image(), overwrite=True)
        layer.bind_key("Escape", lambda lyr: self._cancel_drawing(), overwrite=True)
        layer.bind_key("Delete", lambda lyr: self._delete_under_cursor(), overwrite=True)
        layer.bind_key("Backspace", lambda lyr: self._delete_under_cursor(), overwrite=True)
        layer.bind_key("G", lambda lyr: self.use_glide(), overwrite=True)
        layer.bind_key("P", lambda lyr: self.use_points(), overwrite=True)
        layer.bind_key("B", lambda lyr: self.use_brush(), overwrite=True)
        layer.bind_key("E", lambda lyr: self.use_erase(), overwrite=True)
        layer.bind_key("C", lambda lyr: self.use_click(), overwrite=True)
        layer.bind_key("Z", lambda lyr: self.use_pan(), overwrite=True)
        layer.bind_key("F", lambda lyr: self.fill_holes(), overwrite=True)
        layer.bind_key("D", lambda lyr: self.use_lane(), overwrite=True)
        layer.bind_key("[", lambda lyr: self._cycle_class(-1), overwrite=True)
        layer.bind_key("]", lambda lyr: self._cycle_class(1), overwrite=True)
        for index in range(9):
            layer.bind_key(str(index + 1), lambda lyr, i=index: self._select_class_index(i), overwrite=True)
        for key in ("L", "M", "X", "-", "="):
            layer.bind_key(key, _block_label_shortcut, overwrite=True)

    def _track_cursor(self, layer, event):
        try:
            self._cursor = self._pointer(layer, event)
        except Exception:
            self._cursor = None

    def _cancel_drawing(self):
        self._cancel_glide()
        if self._tool == "lane":
            self.use_glide()
            self._set_status("Lane marking cancelled.")
        if self.labels_layer is not None:
            self._drop_polygon_point(self.labels_layer)

    def _select_class_index(self, index: int):
        if 0 <= index < self.class_list.count():
            self.class_list.setCurrentRow(index)
            self._set_status(f"New cells will be {self.current_class()}.")

    def _cycle_class(self, step: int):
        count = self.class_list.count()
        if not count:
            return
        row = self.class_list.currentRow()
        if row < 0:
            row = 0
        self._select_class_index((row + step) % count)

    def _delete_under_cursor(self):
        if self.labels_layer is None or self._cursor is None:
            return
        data = np.asarray(self.labels_layer.data)
        row, col = int(round(float(self._cursor[0]))), int(round(float(self._cursor[1])))
        if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
            return
        lab = int(data[row, col])
        if lab > 0:
            self.delete_cell(lab)

    def _image_index(self):
        if self.image_path is None:
            return [], -1
        images = io.list_images(self.image_path.parent)
        try:
            index = [p.resolve() for p in images].index(self.image_path.resolve())
        except (OSError, ValueError):
            index = -1
        return images, index

    def previous_image(self):
        self._step_image(-1)

    def next_image(self):
        self._step_image(1)

    def _step_image(self, step: int):
        images, index = self._image_index()
        if index < 0:
            self._set_status("Open an image from a folder to move between them.")
            return
        target = index + step
        if target < 0 or target >= len(images):
            self._set_status("No further images in this folder.")
            return
        if self.confirm_discard():
            self.load_image(images[target])

    def show_recent(self):
        paths = [p for p in recent_images() if Path(p).is_file()]
        if not paths:
            self._set_status("No recent images yet.")
            return
        menu = QMenu(self)
        for path in paths:
            menu.addAction(Path(path).name, lambda checked=False, p=path: self._open_recent(p))
        menu.exec(QCursor.pos())

    def _open_recent(self, path: str):
        if self.confirm_discard():
            self.load_image(path)

    def _quiet_viewer(self):
        """Hide napari's layer list, layer controls and shortcut bar."""
        try:
            window = self.viewer.window._qt_window
            window._qt_viewer.dockLayerList.setVisible(False)
            window._qt_viewer.dockLayerControls.setVisible(False)
            window.statusBar().hide()
        except Exception:
            pass
        layer = self.labels_layer
        if layer is not None:
            try:
                layer.help = "Glide or click around a cell. Right-click a cell to set its class."
            except Exception:
                pass

    def _set_status(self, text: str):
        self.status.setText(text)
        try:
            self.viewer.status = text
        except Exception:
            pass

    def _mark_primary(self, button: QPushButton, primary: bool):
        button.setObjectName("primary" if primary else "")
        button.style().unpolish(button)
        button.style().polish(button)

    # ----------------------------------------------------------------------- image
    def dragEnterEvent(self, event):  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 (Qt naming)
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in io.IMAGE_EXTS and path.is_file():
                if self.confirm_discard():
                    self.load_image(path)
                return

    def open_image(self):
        if not self.confirm_discard():
            return
        start = str(self.image_path.parent) if self.image_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, IMAGE_FILTER)
        if path:
            self.load_image(path)

    def open_folder(self):
        if not self.confirm_discard():
            return
        start = str(self.image_path.parent) if self.image_path else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", start)
        if not folder:
            return
        images = io.list_images(Path(folder))
        if not images:
            self._set_status("That folder has no images.")
            return
        self.load_image(images[0])

    def _refresh_folder_list(self):
        images = []
        current = -1
        if self.image_path is not None:
            images = io.list_images(self.image_path.parent)
            try:
                current = [p.resolve() for p in images].index(self.image_path.resolve())
            except (OSError, ValueError):
                current = -1
        self._folder_images = images
        self.folder_list.blockSignals(True)
        self.folder_list.clear()
        for path in images:
            self.folder_list.addItem(path.name)
        if 0 <= current < self.folder_list.count():
            self.folder_list.setCurrentRow(current)
        self.folder_list.blockSignals(False)

    def _open_folder_item(self, item):
        row = self.folder_list.row(item)
        if row < 0 or row >= len(self._folder_images):
            return
        target = self._folder_images[row]
        if self.image_path is not None and target.resolve() == self.image_path.resolve():
            return
        if self.confirm_discard():
            self.load_image(target)

    def _select_segment_key(self, key: str):
        for index in range(self.segment_choice.count()):
            if self.segment_choice.itemData(index) == key:
                self.segment_choice.setCurrentIndex(index)
                return

    def _prefer_infected_spots(self, path: Path):
        if path.suffix.lower() in (".tif", ".tiff"):
            self._select_segment_key("spots")

    def label_infected_now(self) -> bool:
        """One-step path: Infected spots, then Segment."""
        self._select_segment_key("spots")
        return self.segment_image()

    def _toggle_infected_visible(self):
        self._set_class_visible("infected", self.show_infected_btn.isChecked())

    def _toggle_duplicate_visible(self):
        self._set_class_visible("duplicate", self.show_duplicate_btn.isChecked())

    def _set_class_visible(self, name: str, visible: bool):
        if visible:
            self._hidden_classes.discard(name)
        else:
            self._hidden_classes.add(name)
        self._recolor()
        shown = "shown" if visible else "hidden"
        self._set_status(f"{name.capitalize()} cells {shown}.")

    def find_selected_pair(self):
        lab = self._label_under_cursor()
        if lab <= 0:
            self._set_status("Point at an infected or duplicate cell, then click Find pair.")
            return
        self.show_pair(lab)

    def _label_under_cursor(self) -> int:
        if self.labels_layer is None or self._cursor is None:
            return 0
        data = np.asarray(self.labels_layer.data)
        row, col = int(round(float(self._cursor[0]))), int(round(float(self._cursor[1])))
        if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
            return 0
        return int(data[row, col])

    def show_pair(self, lab: int):
        partner = int(self._spot_pairs.get(int(lab), 0))
        if partner <= 0 or self.labels_layer is None:
            self._set_status("No duplicate pair for that cell.")
            return
        data = np.asarray(self.labels_layer.data)
        centers = []
        for value in (int(lab), partner):
            ys, xs = np.nonzero(data == value)
            if ys.size:
                centers.append((float(ys.mean()), float(xs.mean())))
        if not centers:
            self._set_status("That pair is no longer on the image.")
            return
        row = sum(y for y, _x in centers) / len(centers)
        col = sum(x for _y, x in centers) / len(centers)
        try:
            camera = self.viewer.scene.camera
            center = list(camera.center)
            if len(center) >= 2:
                center[-2] = row
                center[-1] = col
                camera.center = tuple(center)
        except Exception:
            pass
        name = self.classes.get(int(lab), "cell")
        other = self.classes.get(partner, "cell")
        self._set_status(f"Centered on {name} #{lab} and {other} #{partner}.")

    def show_help(self):
        box = QMessageBox(self)
        box.setWindowTitle("Ani help")
        box.setIcon(QMessageBox.Information)
        box.setText(
            "1. Open an image or a folder of scans.\n"
            "2. Label now for infected cells, or Find lanes for gels/blots.\n"
            "3. Keep or Reject the draft, then Generate or Report.\n"
            "4. Batch report builds CSV + profile PNGs for the whole folder.\n"
            "5. Share package / Label Studio for teammates. Large images open as an overview."
        )
        from .machine import summary as machine_summary

        box.setInformativeText(
            "Infected = red, duplicates = green. COCO is included in Share package.\n\n"
            + machine_summary()
        )
        logs = box.addButton("Open logs", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is logs:
            self.open_logs()

    def open_logs(self):
        from .safety import data_dir

        folder = data_dir() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(folder))  # noqa: S606 — Windows open folder
            self._set_status(f"Opened log folder: {folder}")
        except Exception as exc:
            self._set_status(f"Log folder: {folder} ({exc})")

    def load_image(self, path):
        path = Path(path)
        try:
            from .bigimage import read_for_display
            from .cache import IMAGE_CACHE

            cached = IMAGE_CACHE.get(path)
            if cached is not None:
                raw, big = cached, {"overview": False, "note": "", "scale": 1.0, "full_shape": tuple(cached.shape)}
            else:
                raw, big = read_for_display(path)
                IMAGE_CACHE.put(path, raw)
            shown, info = prepare_for_display(raw)
            if big.get("note"):
                info["note"] = ((info.get("note") or "") + " " + big["note"]).strip()
            info["overview"] = bool(big.get("overview"))
            info["overview_scale"] = float(big.get("scale") or 1.0)
        except Exception as e:
            self._set_status(f"Could not open {path.name}: {e}")
            return
        spatial = shown.shape[:2]
        labels = np.zeros(spatial, np.uint16)
        loaded_names: list[str] = []
        loaded_classes: dict[int, str] = {}
        seg_path = path.with_name(f"{path.stem}_seg.npy")
        loaded_from = None
        notice = ""
        if seg_path.exists():
            try:
                masks, loaded_names, loaded_classes = read_seg(seg_path)
            except Exception as e:
                masks = None
                notice = f"Could not read {seg_path.name}: {e}"
            if masks is not None and masks.shape == spatial:
                labels = io.to_label_dtype(masks)
                loaded_from = seg_path
            elif masks is not None:
                notice = f"{seg_path.name} does not match this image, so it was not loaded."
        recovered = load_recovery(path)
        restored = False
        if recovered is not None and recovered["masks"].shape == spatial:
            if np.array_equal(recovered["masks"], labels):
                clear_recovery(path)
            else:
                labels = io.to_label_dtype(recovered["masks"])
                loaded_names = [str(n) for n in recovered["names"]]
                loaded_classes = {int(k): str(v) for k, v in recovered["classes"].items()}
                restored = True
                notice = "Restored unsaved cells from the last session."
        for name in loaded_names:
            if name and name not in self.class_names:
                self.class_names.append(name)
        self._refresh_class_list(self.current_class() if self.class_list.count() else self.class_names[0])

        self._poly_pts = []
        self._poly_click = None
        self._clear_preview()
        self._clear_poly()
        self._rois = []
        self._lane_roi = None
        self._density_mask = None
        self._density_kind = None
        self._lane_layer = None
        self._surface_layer = None
        self._spot_pairs = {}
        self._segment_draft = False
        self._gray_cache = None
        self.accept_draft_btn.hide()
        self.reject_draft_btn.hide()
        try:
            if self.viewer.dims.ndisplay != 2:
                self.viewer.dims.ndisplay = 2
        except Exception:
            pass
        self._volume_backup = None
        self._internal = True
        try:
            self.viewer.layers.clear()
            self.image_layer = None
            image = self.viewer.add_image(shown, name=path.name, rgb=shown.ndim == 3)
            self.image_layer = image
            if shown.ndim == 2:
                try:
                    image.contrast_limits = io.contrast_limits(shown)
                except Exception:
                    pass
            layer = self.viewer.add_labels(labels, name="cells", opacity=self.opacity_slider.value() / 100)
            layer.contour = 1
            try:
                layer._history_limit = 1
                layer._reset_history()
            except Exception:
                pass
            layer.events.data.connect(self._on_data)
            layer.mouse_drag_callbacks.append(self._on_glide)
            layer.mouse_drag_callbacks.append(self._on_mouse)
            layer.mouse_move_callbacks.append(self._track_cursor)
            layer.mouse_double_click_callbacks.append(self._close_outline)
            self._arm_canvas_keys(layer)
            self.labels_layer = layer
            self.viewer.layers.selection.active = layer
        except Exception as exc:
            log_error("open image", exc)
            self._set_status(f"Could not show {path.name}: {exc}")
            return
        finally:
            self._internal = False

        self._segment_token += 1
        self.image_path = path
        self._display_image = np.array(shown, copy=True)
        self.display_info = info
        self.classes = {int(k): v for k, v in loaded_classes.items()}
        self._known = {int(i) for i in tools.label_ids(labels)}
        default = self.current_class()
        for i in self._known:
            self.classes.setdefault(i, default)
        self._snapshot = (np.copy(labels), dict(self.classes))
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._rev = 1 if restored else 0
        self._saved_rev = 0
        self.dirty = restored
        self._restored = restored
        height, width = spatial
        saved, total = folder_progress(path)
        progress = f"  ·  {saved} of {total} saved" if total else ""
        self.file_label.setText(f"{path.name}  ·  {width}×{height}{progress}")
        self._remember_review()
        self.viewer.title = f"Ani — {path.name}"
        try:
            remember_image(path)
        except Exception as exc:
            log_error("recent images", exc)
        self.labels_layer.selected_label = int(tools.next_free_label(labels))
        self._activate_tool()
        self._prefer_infected_spots(path)
        self._refresh_folder_list()
        self._recolor()
        self._refresh_buttons()
        if self._density_dialog is not None and self._density_dialog.isVisible():
            self._density_dialog.refresh()
        self._quiet_viewer()
        try:
            self.viewer.reset_view()
        except Exception:
            pass
        self._prefetch_neighbors()
        self._prefetch_gray()
        if notice:
            self._set_status(notice)
        elif loaded_from is not None:
            self._set_status(f"Opened {path.name}. Loaded {len(self._known)} cells from {loaded_from.name}.")
        elif path.suffix.lower() in (".tif", ".tiff"):
            self._set_status(f"Opened {path.name}. Click Label now to find infected cells and duplicates.")
        elif info.get("note"):
            self._set_status(f"Opened {path.name}. {info['note']} Draw around a cell, then double-click to close.")
        else:
            self._set_status(f"Opened {path.name}. Draw around a cell, then double-click to close.")

    def _prefetch_gray(self):
        """Convert to brightness in the background so Label now is ready immediately."""
        if self._display_image is None:
            return
        if self._gray_job is not None and self._gray_job.isRunning():
            return
        job = _GrayJob(self._display_image)
        job.succeeded.connect(self._on_gray_ready)
        self._gray_job = job
        job.start()

    def _on_gray_ready(self, gray):
        self._gray_job = None
        if self._display_image is None:
            return
        # Drop the cache if the user already switched images.
        if gray is None or getattr(gray, "shape", None) is None:
            return
        if tuple(gray.shape[:2]) != tuple(np.asarray(self._display_image).shape[:2]):
            return
        self._gray_cache = gray

    def _prefetch_neighbors(self):
        """Warm the next image in this folder so Next feels instant."""
        images, index = self._image_index()
        if index < 0:
            return
        target = None
        if index + 1 < len(images):
            target = images[index + 1]
        elif index - 1 >= 0:
            target = images[index - 1]
        if target is None:
            return
        from .cache import IMAGE_CACHE

        if IMAGE_CACHE.get(target) is not None:
            return
        if self._prefetch_job is not None and self._prefetch_job.isRunning():
            return
        job = _PrefetchJob(target)
        self._prefetch_job = job
        job.start()

    # ----------------------------------------------------------------------- classes
    def current_class(self) -> str:
        item = self.class_list.currentItem()
        if item is not None:
            return item.text()
        return self.class_names[0]

    def _refresh_class_list(self, select: str | None = None):
        name = select or (self.current_class() if self.class_list.count() else self.class_names[0])
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, cls in enumerate(self.class_names):
            color = CLASS_COLORS.get(cls, PALETTE[i % len(PALETTE)])
            self.class_list.addItem(QListWidgetItem(_swatch(color), cls))
        row = self.class_names.index(name) if name in self.class_names else 0
        if self.class_list.count():
            self.class_list.setCurrentRow(row)
        self.class_list.blockSignals(False)

    def add_class(self, name: str | None = None):
        if name is None:
            name, ok = QInputDialog.getText(self, "Add class", "Class name")
            if not ok:
                return
        name = " ".join(str(name).split())
        if not name:
            return
        if name not in self.class_names:
            self.class_names.append(name)
        self._refresh_class_list(name)
        self._recolor()
        self._set_status(f"New cells will be '{name}'.")

    def _color_of(self, name: str) -> str:
        if name in CLASS_COLORS:
            return CLASS_COLORS[name]
        if name in self.class_names:
            return PALETTE[self.class_names.index(name) % len(PALETTE)]
        return PALETTE[0]

    def _recolor(self):
        layer = self.labels_layer
        if layer is None:
            return
        try:
            from napari.utils.colormaps import DirectLabelColormap
        except Exception:
            return
        current = self._color_of(self.current_class())
        colors: dict = {None: current, 0: "transparent"}
        for lab, cls in self.classes.items():
            if str(cls) in self._hidden_classes:
                colors[int(lab)] = "transparent"
            else:
                colors[int(lab)] = self._color_of(cls)
        nxt = int(layer.selected_label)
        colors[nxt] = current
        layer.colormap = DirectLabelColormap(color_dict=colors)

    # ----------------------------------------------------------------------- drawing
    def _activate_tool(self):
        {
            "glide": self.use_glide, "points": self.use_points, "pan": self.use_pan,
            "brush": self.use_brush, "erase": self.use_erase, "click": self.use_click,
            "lane": self.use_lane,
        }[self._tool]()

    def use_draw(self):
        self.use_points()

    def use_glide(self):
        self._tool = "glide"
        self.glide_btn.setChecked(True)
        self.draw_hint.setText("Drag around a spot. Pan moves the image. Ctrl+click places corners.")
        self._arm_labels(pan=False)

    def use_points(self):
        self._cancel_glide()
        self._tool = "points"
        self.points_btn.setChecked(True)
        self.draw_hint.setText("Click corners, double-click to close. Right-click removes the last point.")
        layer = self.labels_layer
        if layer is None:
            return
        self.viewer.layers.selection.active = layer
        layer.selected_label = int(tools.next_free_label(layer.data))
        try:
            layer.mode = "polygon"
        except ValueError:
            layer.mode = "paint"
        self._allow_drag_pan()
        self._recolor()
        self._quiet_viewer()

    def use_pan(self):
        self._cancel_glide()
        self._tool = "pan"
        self.pan_btn.setChecked(True)
        self.draw_hint.setText("Drag moves the image. The wheel zooms. Ctrl+click places corners.")
        self._arm_labels(pan=True)

    def use_brush(self):
        self._cancel_glide()
        self._tool = "brush"
        self.brush_btn.setChecked(True)
        self.draw_hint.setText("Drag to paint. Start on a cell to extend it. Pan moves the image.")
        self._arm_labels(pan=False)

    def use_erase(self):
        self._cancel_glide()
        self._tool = "erase"
        self.erase_btn.setChecked(True)
        self.draw_hint.setText("Drag to erase. Ctrl+Z puts it back. Pan moves the image.")
        self._arm_labels(pan=False)

    def use_click(self):
        self._cancel_glide()
        self._tool = "click"
        self.click_btn.setChecked(True)
        self.draw_hint.setText("Click one object. Other outlines stay. Pan moves the image.")
        self._arm_labels(pan=False)

    def use_lane(self):
        """Keep the normal drawing tools. The outlines become the areas to measure."""
        if self.labels_layer is None or self._display_image is None:
            self._set_status("Open an image first.")
            return
        if self._tool == "lane":
            self._tool = "glide"
        if self._surface_layer is not None or int(getattr(self.viewer.dims, "ndisplay", 2)) != 2:
            self.show_density_2d()
        self.use_glide()
        self.draw_hint.setText("Drag with Glide around each spot. Points and Brush work too.")
        self._set_status("Mark each area the same way you annotate. Click Generate when the spots are done.")

    def generate_densitogram(self):
        """Graph and table for every outlined spot. Drawing itself does not calculate anything."""
        if self._display_image is None:
            self._set_status("Open an image first.")
            return
        self.flush()
        outlines = self._rois_from_outlines()
        if outlines:
            self._rois = outlines
        self._refresh_buttons()
        if not self._rois:
            self._set_status("Draw spots, or click Find lanes for a gel. Generate uses those areas.")
            return
        self.open_densitogram()
        self._set_status(
            f"{len(self._rois)} areas ready. Use Export CSV, Report, or Save package for the table."
        )

    def open_densitogram(self):
        from .density_ui import DensityDialog

        if self._display_image is None or not self._rois:
            self._set_status("Mark an ROI first.")
            return
        if self._density_dialog is None:
            self._density_dialog = DensityDialog(self)
        self._density_dialog.refresh()
        self._density_dialog.show()
        self._density_dialog.raise_()
        self._density_dialog.activateWindow()

    def show_density_3d(self):
        """Turn each outlined spot on its own. Drawing stays off while the view turns."""
        if self._display_image is None:
            self._set_status("Open an image first.")
            return
        self.flush()
        outlines = self._rois_from_outlines()
        if outlines:
            self._rois = outlines
        if not self._rois:
            self._set_status("Draw a spot or Find lanes first. 3D uses those areas.")
            return
        from .density import combine_surfaces, surface_mesh

        parts = []
        for item in self._rois:
            parts.append(surface_mesh(
                self._display_image, roi=item["roi"], dark_bands=self._density_dark,
                mask=self._roi_mask(item), max_side=96,
            ))
        vertices, faces, values = combine_surfaces(parts)
        if faces.size == 0:
            self._set_status("Those spots are too small to show in 3D.")
            return
        self._cancel_glide()
        self._drop_surface()
        try:
            layer = self.viewer.add_surface(
                (vertices, faces, values), name="Density", shading="smooth", colormap="gray",
            )
        except Exception as exc:
            log_error("density surface", exc)
            self._set_status(f"Could not show the 3D surface: {exc}")
            return
        self._surface_layer = layer
        self._viewing_3d = True
        self._shrink_layers_for_volume()
        self.viewer.dims.ndisplay = 3
        # The camera turns only while the active layer allows a drag.
        # Drawing stays off through _viewing_3d, so this drag is only rotation.
        try:
            layer.mouse_pan = True
        except Exception:
            pass
        if self.labels_layer is not None:
            self.labels_layer.mouse_pan = True
        self.viewer.layers.selection.active = layer
        try:
            self.viewer.scene.camera.mouse_pan = True
            self.viewer.scene.camera.mouse_zoom = True
            self.viewer.scene.camera.angles = (22.0, 28.0, 0.0)
        except Exception:
            pass
        self._set_status("Drag the pointer to turn the view in any direction. Normal puts the picture back.")

    def show_density_2d(self):
        """Leave 3D and put the picture back in its usual flat position."""
        self._viewing_3d = False
        self._drop_surface()
        try:
            if int(self.viewer.dims.ndisplay) == 3:
                self.viewer.scene.camera.angles = (0.0, 0.0, 0.0)
            self.viewer.dims.ndisplay = 2
            self.viewer.dims.order = tuple(range(int(self.viewer.dims.ndim)))
        except Exception:
            pass
        self._restore_volume_layers()
        try:
            self.viewer.scene.camera.angles = (0.0, 0.0, 0.0)
            self.viewer.reset_view()
        except Exception:
            pass
        self._set_status("Back to the normal view.")
        if self.labels_layer is not None and self._tool in ("glide", "points", "brush", "erase", "click", "pan"):
            self._activate_tool()

    def _volume_texture_limit(self) -> int:
        """Largest cube this computer can draw in 3D."""
        try:
            from napari._vispy.utils.gl import get_max_texture_sizes

            _flat, volume = get_max_texture_sizes()
            if volume:
                return int(volume)
        except Exception:
            pass
        return 2048

    def _guard_display_button(self):
        """The viewer's own 3D button must fit the picture before turning."""
        try:
            button = self.viewer.window._qt_viewer.viewerButtons.ndisplayButton
        except Exception:
            return
        try:
            button.clicked.disconnect()
        except Exception:
            pass
        button.clicked.connect(self._toggle_display_mode)

    def _toggle_display_mode(self, *_args):
        if int(self.viewer.dims.ndisplay) == 3:
            if self._viewing_3d:
                self.show_density_2d()
            else:
                self.viewer.dims.ndisplay = 2
                self._restore_volume_layers()
            return
        self._shrink_layers_for_volume()
        self.viewer.dims.ndisplay = 3

    def _shrink_layers_for_volume(self):
        """Fit the picture inside the 3D texture limit without moving it."""
        if self._volume_backup is not None or self.image_layer is None or self._fitting_volume:
            return
        image = np.asarray(self.image_layer.data)
        if image.ndim < 2:
            return
        limit = self._volume_texture_limit()
        longest = max(int(image.shape[0]), int(image.shape[1]))
        if longest <= limit:
            return
        factor = limit / longest
        height = max(1, int(round(image.shape[0] * factor)))
        width = max(1, int(round(image.shape[1] * factor)))
        labels = None if self.labels_layer is None else np.asarray(self.labels_layer.data)
        self._fitting_volume = True
        self._internal = True
        try:
            self._volume_backup = {
                "image": np.array(image, copy=True),
                "image_scale": tuple(float(v) for v in self.image_layer.scale),
                "labels": None if labels is None else np.array(labels, copy=True),
                "labels_scale": None if self.labels_layer is None else tuple(float(v) for v in self.labels_layer.scale),
            }
            small = _fit_plane(image, height, width)
            self.image_layer.data = small
            scale = [float(v) for v in self._volume_backup["image_scale"]]
            scale[0] *= image.shape[0] / small.shape[0]
            scale[1] *= image.shape[1] / small.shape[1]
            self.image_layer.scale = tuple(scale)
            if self.labels_layer is not None and labels is not None and labels.ndim >= 2:
                small_labels = _fit_plane(labels, height, width)
                self.labels_layer.data = small_labels
                lscale = [float(v) for v in self._volume_backup["labels_scale"]]
                lscale[0] *= labels.shape[0] / small_labels.shape[0]
                lscale[1] *= labels.shape[1] / small_labels.shape[1]
                self.labels_layer.scale = tuple(lscale)
        finally:
            self._internal = False
            self._fitting_volume = False

    def _restore_volume_layers(self):
        backup = self._volume_backup
        if backup is None or self._fitting_volume:
            return
        self._volume_backup = None
        self._fitting_volume = True
        self._internal = True
        try:
            if self.image_layer is not None:
                self.image_layer.data = backup["image"]
                self.image_layer.scale = backup["image_scale"]
            if self.labels_layer is not None and backup["labels"] is not None:
                self.labels_layer.data = backup["labels"]
                self.labels_layer.scale = backup["labels_scale"]
        finally:
            self._internal = False
            self._fitting_volume = False

    def _drop_surface(self):
        layer = self._surface_layer
        self._surface_layer = None
        if layer is not None and layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)

    def _set_brush_size(self, value: int):
        self.brush_label.setText(f"Brush {int(value)} px")

    def _set_opacity(self, value: int):
        self.opacity_label.setText(f"Fill {int(value)}%")
        if self.labels_layer is not None:
            self.labels_layer.opacity = int(value) / 100

    def fill_holes(self):
        if self.labels_layer is None:
            return
        self.flush()
        filled, count = fill_label_holes(np.asarray(self.labels_layer.data))
        if count == 0:
            self._set_status("No holes inside cells.")
            return
        self.labels_layer.data = filled
        self.flush()
        self._set_status(f"Filled {count} hole{'s' if count != 1 else ''}.")

    def check_outlines(self):
        if self.labels_layer is None:
            self._set_status("Open an image first.")
            return
        self.flush()
        self._review_timer.stop()
        self._run_review()
        self._set_status(review_message(self._review))

    def fit_view(self):
        try:
            self.viewer.reset_view()
        except Exception:
            pass

    def auto_contrast(self):
        layer = self.image_layer
        shown = self._display_image
        if layer is None or shown is None:
            self._set_status("Open an image first.")
            return
        try:
            if shown.ndim == 2:
                layer.contrast_limits = io.contrast_limits(shown)
            else:
                layer.contrast_limits = [
                    io.contrast_limits(shown[..., i]) for i in range(min(3, shown.shape[-1]))
                ]
        except Exception as exc:
            log_error("contrast", exc)
            self._set_status("Could not set the contrast.")
            return
        self._set_status("Contrast set from the image.")

    def _pixel_size(self) -> str:
        text = self.pixel_edit.text().strip().replace(",", ".")
        if not text:
            return ""
        try:
            value = float(text)
        except ValueError:
            return ""
        if not np.isfinite(value) or value <= 0:
            return ""
        return str(value)

    def _arm_labels(self, pan: bool):
        layer = self.labels_layer
        if layer is None:
            return
        self.viewer.layers.selection.active = layer
        layer.mode = "pan_zoom"
        if pan:
            self._allow_drag_pan()
        else:
            layer.mouse_pan = False
            try:
                self.viewer.scene.camera.mouse_pan = False
            except Exception:
                pass
        self._recolor()
        self._quiet_viewer()

    def _allow_drag_pan(self):
        """Left-button dragging moves the image horizontally and vertically."""
        layer = self.labels_layer
        if layer is not None:
            layer.mouse_pan = True
        try:
            self.viewer.scene.camera.mouse_pan = True
        except Exception:
            pass

    def _pointer(self, layer, event) -> np.ndarray:
        """The image pixel under the cursor, in row, column order."""
        coordinates = None
        try:
            from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate

            coordinates = mouse_event_to_labels_coordinate(layer, event)
        except Exception:
            coordinates = None
        if coordinates is not None:
            data = np.asarray(coordinates, float).ravel()
            if data.size >= 2 and np.all(np.isfinite(data[-2:])):
                return data[-2:]
        world = np.asarray(getattr(event, "position", (0.0, 0.0)), float)
        try:
            data = np.asarray(layer.world_to_data(world), float)
        except Exception:
            data = world
        return np.asarray(data, float).ravel()[-2:]

    def _match_labels_space(self, shapes) -> None:
        """Keep a preview on the same pixels as the image."""
        labels = self.labels_layer
        if labels is None or shapes is None:
            return
        try:
            shapes.scale = tuple(labels.scale)
            shapes.translate = tuple(labels.translate)
        except Exception:
            pass

    def _on_glide(self, layer, event):
        if self._viewing_3d or int(getattr(self.viewer.dims, "ndisplay", 2)) == 3:
            return
        if getattr(event, "button", None) != 1 or getattr(event, "type", "") != "mouse_press":
            return
        mods = getattr(event, "modifiers", ()) or ()
        if self._tool == "glide" and "Control" not in mods:
            yield from self._trace_glide(layer, event)
            return
        if self._tool in ("brush", "erase") and "Control" not in mods and "Shift" not in mods:
            yield from self._stroke(layer, event, erase=self._tool == "erase")
            return
        if self._tool == "click" and "Control" not in mods and "Shift" not in mods:
            yield from self._click_segment(layer, event)
            return
        if self._tool == "lane" and "Control" not in mods and "Shift" not in mods:
            yield from self._trace_lane(layer, event)
            return
        if "Control" in mods and self._tool != "points":
            yield from self._click_annotate(layer, event)
            return
        # A plain left-drag moves the image. A click that stays still can still add a corner.
        if self._tool != "points":
            return
        start = np.array(getattr(event, "pos", (0.0, 0.0)), float)
        yield
        while event.type == "mouse_move":
            pos = np.array(getattr(event, "pos", start), float)
            if np.hypot(*(pos - start)) > 4:
                self._drop_polygon_point(layer)
            yield

    def _stroke(self, layer, event, erase: bool):
        """One brush or erase drag is a single undo step, painted only where the brush touches."""
        event.handled = True
        self._stroking = True
        self._brush_last = None
        self._stroke_data = np.asarray(layer.data)
        point = self._pointer(layer, event)
        value = 0 if erase else self._brush_value(point)
        begin = getattr(layer, "_begin_stroke", None)
        if begin is not None:
            begin()
        self._paint_at(point, value)
        try:
            yield
            while event.type == "mouse_move":
                self._paint_at(self._pointer(layer, event), value)
                yield
        finally:
            commit = getattr(layer, "_commit_stroke", None)
            if commit is not None:
                commit()
            try:
                layer.refresh(extent=False)
            except Exception:
                pass
            self._stroking = False
            self._stroke_data = None
            self._brush_last = None
            before = self._rev
            self.flush()
        if self._rev == before:
            self._set_status("That stroke did not change the outlines.")
        elif erase:
            self._set_status("Erased. Ctrl+Z puts it back.")
        else:
            self._set_status("Brush stroke added. Right-click it to set the class.")

    def _brush_value(self, point: np.ndarray) -> int:
        data = self._stroke_data if self._stroke_data is not None else np.asarray(self.labels_layer.data)
        row, col = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1] and int(data[row, col]) > 0:
            return int(data[row, col])
        return int(tools.next_free_label(data))

    def _paint_at(self, point: np.ndarray, value: int):
        """Update just the brush pixels, and redraw at most about 60 times a second."""
        layer = self.labels_layer
        if layer is None:
            return
        radius = max(0.5, self.brush_slider.value() / 2)
        start = point if self._brush_last is None else self._brush_last
        rr, cc = brush_indices(np.asarray(layer.data).shape, start, point, radius)
        self._brush_last = np.asarray(point, float)
        if rr.size == 0:
            return
        now = time.monotonic()
        refresh = now - self._paint_drawn >= 0.016
        if refresh:
            self._paint_drawn = now
        layer.data_setitem((rr, cc), int(value), refresh=refresh)

    def _trace_lane(self, layer, event):
        """Drag a band, or click one stained cell. Nothing else is measured."""
        event.handled = True
        start = self._pointer(layer, event)
        yield
        while event.type == "mouse_move":
            point = self._pointer(layer, event)
            if np.hypot(*(np.asarray(point, float) - np.asarray(start, float))) >= 4:
                self._preview_lane(start, point)
            yield
        end = self._pointer(layer, event)
        if np.hypot(*(np.asarray(end, float) - np.asarray(start, float))) < 4:
            self._select_stained_cell(end)
            return
        roi = self._roi_from_drag(start, end)
        if roi is None:
            self._set_status("Drag a wider rectangle around the band, or click a stained cell.")
            return
        self._remember_roi(roi, None, "band")
        rows, cols = roi[2] - roi[0], roi[3] - roi[1]
        self._set_status(f"ROI {len(self._rois)} marked ({cols}×{rows} px). Click Generate for the graph and peak table.")

    def _roi_from_drag(self, start, end):
        if self.labels_layer is None:
            return None
        shape = np.asarray(self.labels_layer.data).shape
        r0, r1 = sorted((int(round(float(start[0]))), int(round(float(end[0])))))
        c0, c1 = sorted((int(round(float(start[1]))), int(round(float(end[1])))))
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(int(shape[0]), r1 + 1), min(int(shape[1]), c1 + 1)
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        return (r0, c0, r1, c1)

    def _select_stained_cell(self, point):
        """Measure one outlined cell, and only the pixels inside that outline."""
        if self.labels_layer is None:
            return
        labels = np.asarray(self.labels_layer.data)
        row, col = int(round(float(point[0]))), int(round(float(point[1])))
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]) or int(labels[row, col]) == 0:
            self._set_status("Click a stained cell, or drag a rectangle around one band.")
            return
        stain = labels == int(labels[row, col])
        rows, cols = np.nonzero(stain)
        roi = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
        if roi[2] - roi[0] < 4 or roi[3] - roi[1] < 4:
            self._set_status("That stained area is too small to measure.")
            return
        self._remember_roi(roi, stain, "cell")
        name = self.classes.get(int(labels[row, col]), self.class_names[0])
        self._set_status(f"ROI {len(self._rois)} is the stained area of this {name}. Click Generate when you want the numbers.")

    def _remember_roi(self, roi, mask, kind: str):
        self._rois.append({"roi": roi, "mask": mask, "kind": kind})
        self._lane_roi = roi
        self._density_mask = mask
        self._density_kind = kind
        self._show_roi_boxes()
        self._refresh_buttons()

    def _rois_from_outlines(self) -> list[dict]:
        """Each finished outline is one area. Background pixels stay out."""
        if self.labels_layer is None:
            return []
        labels = np.asarray(self.labels_layer.data)
        found = []
        for lab in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
            rows, cols = np.nonzero(labels == lab)
            if rows.size < 16:
                continue
            roi = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
            if roi[2] - roi[0] < 4 or roi[3] - roi[1] < 4:
                continue
            found.append({
                "roi": roi,
                "mask": None,
                "kind": "cell",
                "label": lab,
                "name": self.classes.get(lab, self.class_names[0] if self.class_names else "cell"),
            })
        return found

    def _roi_mask(self, item):
        if item.get("mask") is not None:
            return item["mask"]
        label = item.get("label")
        if label is None or self.labels_layer is None:
            return None
        return np.asarray(self.labels_layer.data) == int(label)

    def _sync_active_roi(self) -> bool:
        """Point the 3D view at the area chosen in the graph, or the last one marked."""
        if not self._rois:
            return False
        index = len(self._rois) - 1
        dialog = self._density_dialog
        choice = getattr(dialog, "roi_choice", None) if dialog is not None else None
        if choice is not None and choice.count():
            chosen = choice.currentIndex()
            if chosen >= 0:
                index = min(chosen, len(self._rois) - 1)
        item = self._rois[index]
        self._lane_roi = item["roi"]
        self._density_mask = self._roi_mask(item)
        self._density_kind = item["kind"]
        return True

    def _prepare_density_selection(self) -> bool:
        """Use the outline shown in the graph, or every outline when the graph is closed."""
        self.flush()
        outlines = self._rois_from_outlines()
        if outlines:
            self._rois = outlines
        if not self._rois:
            return False
        dialog = self._density_dialog
        if dialog is not None and dialog.isVisible() and getattr(dialog, "roi_choice", None) is not None and dialog.roi_choice.count():
            return self._sync_active_roi()
        labels = np.asarray(self.labels_layer.data) if self.labels_layer is not None else None
        ids = [int(item["label"]) for item in self._rois if item.get("label") is not None]
        if labels is None or not ids:
            return self._sync_active_roi()
        mask = np.isin(labels, np.asarray(ids, dtype=labels.dtype))
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return False
        self._lane_roi = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
        self._density_mask = mask
        self._density_kind = "cell"
        return True

    def _preview_lane(self, start, end):
        r0, c0 = float(start[0]), float(start[1])
        r1, c1 = float(end[0]), float(end[1])
        self._show_roi_boxes(np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]], float))

    def _show_roi_boxes(self, draft=None):
        colors = ("#39e0ff", "#ffb020", "#e86bff", "#3ddc97", "#ff5c5c", "#7aa2ff")
        rects = []
        for item in self._rois:
            r0, c0, r1, c1 = item["roi"]
            rects.append(np.array([[r0, c0], [r0, c1 - 1], [r1 - 1, c1 - 1], [r1 - 1, c0]], float))
        if draft is not None:
            rects.append(np.asarray(draft, float))
        if not rects:
            layer = self._lane_layer
            self._lane_layer = None
            if layer is not None and layer in list(self.viewer.layers):
                self.viewer.layers.remove(layer)
            return
        edge = [colors[i % len(colors)] for i in range(len(rects))]
        try:
            if self._lane_layer is None or self._lane_layer not in list(self.viewer.layers):
                self._lane_layer = self.viewer.add_shapes(
                    rects, shape_type="rectangle", edge_width=2, edge_color=edge,
                    face_color=[0, 0, 0, 0], name="ROI",
                )
                self._match_labels_space(self._lane_layer)
                if self.labels_layer is not None:
                    self.viewer.layers.selection.active = self.labels_layer
            else:
                self._lane_layer.data = rects
                self._lane_layer.edge_color = edge
        except Exception:
            self._lane_layer = None

    def _click_segment(self, layer, event):
        """A still click outlines one object. A drag does nothing, so the image can be left alone."""
        event.handled = True
        point = self._pointer(layer, event)
        start = np.array(getattr(event, "pos", (0.0, 0.0)), float)
        dragged = False
        yield
        while event.type == "mouse_move":
            pos = np.array(getattr(event, "pos", start), float)
            if np.hypot(*(pos - start)) > 4:
                dragged = True
            yield
        if not dragged:
            self.segment_at_point(point)

    def _click_annotate(self, layer, event):
        """Ctrl+left-click places a corner. Click the first corner, or double-click, to close."""
        event.handled = True
        start = self._pointer(layer, event)
        start_pos = np.array(getattr(event, "pos", (0.0, 0.0)), float)
        dragged = False
        yield
        while event.type == "mouse_move":
            pos = np.array(getattr(event, "pos", start_pos), float)
            if np.hypot(*(pos - start_pos)) > 4:
                dragged = True
            yield
        if not dragged:
            self._add_outline_point(start, start_pos)

    def _add_outline_point(self, point: np.ndarray, canvas: np.ndarray) -> bool:
        now = time.monotonic()
        previous = self._poly_click
        if (
            previous is not None
            and len(self._poly_pts) >= 3
            and now - previous[0] < 0.45
            and np.hypot(*(canvas - previous[1])) <= 6
        ):
            return self._commit_polygon()
        if len(self._poly_pts) >= 3 and np.hypot(*(point - self._poly_pts[0])) <= 8:
            return self._commit_polygon()
        self._poly_pts.append(np.asarray(point, float))
        self._poly_click = (now, np.asarray(canvas, float))
        self._preview_poly()
        if len(self._poly_pts) >= 3:
            nxt = "Click the first point, or double-click, to close."
        else:
            nxt = "Ctrl+click the next corner."
        self._set_status(f"Corner {len(self._poly_pts)} placed. {nxt} Right-click removes the last corner.")
        return False

    def _close_outline(self, layer, event):
        if getattr(event, "button", None) not in (1, None) or len(self._poly_pts) < 3:
            return
        event.handled = True
        self._commit_polygon()

    def _preview_poly(self):
        if len(self._poly_pts) < 2 or self.labels_layer is None:
            return
        pts = np.asarray(self._poly_pts, float)
        if len(pts) >= 3:
            pts = np.vstack([pts, pts[0]])
        color = self._color_of(self.current_class())
        try:
            if self._poly_layer is None or self._poly_layer not in list(self.viewer.layers):
                self._poly_layer = self.viewer.add_shapes(
                    [pts], shape_type="path", edge_width=2, edge_color=color,
                    face_color=[0, 0, 0, 0], name="outline",
                )
                self._match_labels_space(self._poly_layer)
                self.viewer.layers.selection.active = self.labels_layer
                if self._tool in ("pan", "points"):
                    self._allow_drag_pan()
            else:
                self._poly_layer.data = [pts]
                self._poly_layer.edge_color = color
        except Exception:
            self._poly_layer = None

    def _clear_poly(self):
        layer = self._poly_layer
        self._poly_layer = None
        if layer is not None and layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)
        if self.labels_layer is not None and self.labels_layer in list(self.viewer.layers):
            self.viewer.layers.selection.active = self.labels_layer
            if self._tool in ("pan", "points"):
                self._allow_drag_pan()

    def _commit_polygon(self) -> bool:
        """Label the cell enclosed by the Ctrl+click corners."""
        layer = self.labels_layer
        pts = np.asarray(self._poly_pts, float) if self._poly_pts else np.zeros((0, 2))
        self._poly_pts = []
        self._poly_click = None
        self._clear_poly()
        if layer is None or len(pts) < 3:
            return False
        label = int(tools.next_free_label(layer.data))
        layer.paint_polygon(pts, label)
        self.flush()
        if label not in tools.label_ids(layer.data):
            self._set_status("Click at least three corners around the cell.")
            return False
        self._set_status("Cell outlined. Right-click it to change the class.")
        return True

    def _drop_polygon_point(self, layer):
        overlay = getattr(layer, "_overlays", {}).get("polygon") if hasattr(layer, "_overlays") else None
        if overlay is not None and getattr(overlay, "points", None):
            overlay.points = []

    def _trace_glide(self, layer, event):
        event.handled = True
        self._glide_cancel = False
        self._glide_pts = [self._pointer(layer, event)]
        self._preview_glide()
        yield
        while event.type == "mouse_move" and not self._glide_cancel:
            point = self._pointer(layer, event)
            if np.hypot(*(point - self._glide_pts[-1])) >= 1:
                self._glide_pts.append(point)
                self._preview_glide()
            yield
        points = np.asarray(self._glide_pts, float) if self._glide_pts else np.zeros((0, 2))
        cancelled = self._glide_cancel
        self._glide_pts = []
        self._glide_cancel = False
        self._clear_preview()
        if not cancelled:
            self.apply_glide(points)

    def _preview_glide(self):
        if len(self._glide_pts) < 2 or self.labels_layer is None:
            return
        pts = np.asarray(self._glide_pts, float)
        color = self._color_of(self.current_class())
        try:
            if self._glide_layer is None or self._glide_layer not in list(self.viewer.layers):
                self._glide_layer = self.viewer.add_shapes(
                    [pts], shape_type="path", edge_width=2, edge_color=color,
                    face_color=[0, 0, 0, 0], name="glide",
                )
                self._match_labels_space(self._glide_layer)
                self.viewer.layers.selection.active = self.labels_layer
            else:
                self._glide_layer.data = [pts]
                self._glide_layer.edge_color = color
        except Exception:
            self._glide_layer = None

    def _clear_preview(self):
        layer = self._glide_layer
        self._glide_layer = None
        self._glide_pts = []
        if layer is not None and layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)
        if self.labels_layer is not None:
            self.viewer.layers.selection.active = self.labels_layer
            if self._tool in ("pan", "points"):
                self._allow_drag_pan()

    def _cancel_glide(self):
        self._glide_cancel = True
        self._poly_pts = []
        self._poly_click = None
        self._clear_preview()
        self._clear_poly()

    def apply_glide(self, points: np.ndarray) -> bool:
        """Smooth a freehand stroke and label the cell it encloses."""
        layer = self.labels_layer
        if layer is None:
            return False
        pts = np.asarray(points, float)
        mask = mask_from_glide(pts, layer.data.shape[-2:])
        if mask is None or int(mask.sum()) < 8:
            self._set_status("Glide around the whole cell, then release.")
            return False
        data = np.array(layer.data, copy=True)
        data[mask] = int(tools.next_free_label(data))
        layer.data = data
        self.flush()
        gap = float(np.hypot(*(pts[0] - pts[-1]))) if len(pts) > 1 else 0
        length = float(np.hypot(*(np.diff(pts, axis=0).T)).sum()) if len(pts) > 1 else 0
        if gap > 24 and gap > 0.3 * max(length, 1):
            self._set_status("Outline closed across a gap. Glide all the way around the cell next time.")
        else:
            self._set_status("Cell outlined. Right-click it to change the class.")
        return True

    def _on_data(self, event=None):
        if self._internal or self.labels_layer is None:
            return
        self._timer.start()

    def flush(self):
        """Commit a finished outline as its own undo step and give it the selected class."""
        if self._stroking or self._in_flush or self._internal or self.labels_layer is None or self._snapshot is None:
            return
        self._in_flush = True
        try:
            data = np.asarray(self.labels_layer.data)
            prev = self._snapshot[0]
            if prev.shape == data.shape and np.array_equal(prev, data):
                return
            self._push_undo()
            ids = {int(i) for i in tools.label_ids(data)}
            fresh = self.current_class()
            for i in ids - self._known:
                self.classes.setdefault(i, fresh)
            for i in list(self.classes):
                if i not in ids:
                    self.classes.pop(i, None)
            self._known = ids
            self._snapshot = (np.copy(data), dict(self.classes))
            nxt = int(tools.next_free_label(data))
            if int(self.labels_layer.selected_label) != nxt:
                self.labels_layer.selected_label = nxt
            self._recolor()
            self._remember_review()
            self._refresh_buttons()
            self._backup()
        finally:
            self._in_flush = False

    def _value_at(self, layer, event) -> int:
        try:
            value = layer.get_value(event.position, view_direction=event.view_direction,
                                    dims_displayed=event.dims_displayed, world=True)
        except TypeError:
            value = layer.get_value(event.position, world=True)
        if isinstance(value, tuple):
            value = value[-1]
        return int(value) if value is not None else 0

    def _on_mouse(self, layer, event):
        if getattr(event, "type", "") != "mouse_press" or getattr(event, "button", None) != 2:
            return
        if self._poly_pts:
            self._poly_pts.pop()
            self._poly_click = None
            if len(self._poly_pts) >= 2:
                self._preview_poly()
            else:
                self._clear_poly()
            self._set_status("Last corner removed." if self._poly_pts else "Outline cancelled.")
            return
        overlay = getattr(layer, "_overlays", {}).get("polygon") if hasattr(layer, "_overlays") else None
        if overlay is not None and getattr(overlay, "points", None):
            return  # a boundary is in progress: right-click removes the last point
        self.flush()
        lab = self._value_at(layer, event)
        if lab <= 0 or lab not in self._known:
            self._set_status("Right-click a cell to set its class.")
            return
        self._popup_class(lab)

    def _popup_class(self, lab: int):
        menu = QMenu(self)
        colors = self._colors or {}
        if colors:
            menu.setStyleSheet(
                f"QMenu {{ background: {colors['surface']}; color: {colors['text']}; "
                f"border: 1px solid {colors['accent']}; padding: 4px; }}"
                f"QMenu::item {{ padding: 6px 18px; }}"
                f"QMenu::item:selected {{ background: {colors['accent']}; color: white; }}"
            )
        group = QActionGroup(menu)
        group.setExclusive(True)
        current = self.classes.get(lab, self.class_names[0])
        for name in self.class_names:
            act = QAction(name, menu)
            act.setCheckable(True)
            act.setChecked(name == current)
            group.addAction(act)
            act.triggered.connect(lambda checked=False, n=name, i=lab: self.assign_class(i, n))
            menu.addAction(act)
        menu.addSeparator()
        if int(lab) in self._spot_pairs:
            menu.addAction("Show pair", lambda: self.show_pair(lab))
        menu.addAction("New class…", lambda: self._add_class_for(lab))
        menu.addAction("Delete cell", lambda: self.delete_cell(lab))
        menu.exec(QCursor.pos())

    def _add_class_for(self, lab: int):
        name, ok = QInputDialog.getText(self, "New class", "Class name")
        if not ok:
            return
        name = " ".join(name.split())
        if not name:
            return
        if name not in self.class_names:
            self.class_names.append(name)
            self._refresh_class_list(name)
        self.assign_class(lab, name)

    def assign_class(self, lab: int, name: str):
        self.flush()
        lab = int(lab)
        if lab not in self._known or name not in self.class_names:
            return
        if self.classes.get(lab) == name:
            self._set_status(f"Cell {lab} is '{name}'.")
            return
        self._push_undo()
        self.classes[lab] = name
        self._snapshot = (self._snapshot[0], dict(self.classes))
        self._recolor()
        self._refresh_buttons()
        self._set_status(f"Cell {lab} is '{name}'.")

    def delete_cell(self, lab: int):
        self.flush()
        lab = int(lab)
        if self.labels_layer is None or lab not in self._known:
            return
        self.labels_layer.data = tools.delete_label(np.asarray(self.labels_layer.data), lab)
        self.flush()
        self._set_status(f"Deleted cell {lab}.")

    # ------------------------------------------------------------------------- undo
    def _push_undo(self):
        self.undo_stack.append((np.copy(self._snapshot[0]), dict(self._snapshot[1]), self._rev))
        while len(self.undo_stack) > 50 or self._undo_bytes() > 512 * 1024 * 1024:
            if len(self.undo_stack) <= 1:
                break
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._rev += 1
        self.dirty = self._rev != self._saved_rev

    def _restore(self, data, classes, rev):
        self._in_flush = True
        self._internal = True
        try:
            self._rev = rev
            self.labels_layer.data = np.copy(data)
            self.classes = {int(k): v for k, v in classes.items()}
            self._known = {int(i) for i in tools.label_ids(self.labels_layer.data)}
            self._snapshot = (np.copy(self.labels_layer.data), dict(self.classes))
            self.labels_layer.selected_label = int(tools.next_free_label(self.labels_layer.data))
            self.dirty = self._rev != self._saved_rev
        finally:
            self._internal = False
            self._in_flush = False
        self._recolor()
        self._remember_review()
        self._refresh_buttons()

    def _remember_review(self):
        """Recount holes after the stroke, so drawing itself stays instant."""
        if self.labels_layer is None:
            self._review = {}
            return
        self._review_timer.start()

    def _run_review(self):
        if self._closing or self.labels_layer is None:
            return
        try:
            self._review = review_outlines(np.asarray(self.labels_layer.data), self.classes)
            self.summary.setText(self._summary_text())
        except RuntimeError:
            return

    def undo(self):
        if self.labels_layer is None:
            return
        self.flush()
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return
        self.redo_stack.append((np.copy(self._snapshot[0]), dict(self._snapshot[1]), self._rev))
        data, classes, rev = self.undo_stack.pop()
        self._restore(data, classes, rev)
        self._set_status("Undone.")

    def redo(self):
        if self.labels_layer is None:
            return
        self.flush()
        if not self.redo_stack:
            self._set_status("Nothing to redo.")
            return
        self.undo_stack.append((np.copy(self._snapshot[0]), dict(self._snapshot[1]), self._rev))
        data, classes, rev = self.redo_stack.pop()
        self._restore(data, classes, rev)
        self._set_status("Redone.")

    def _summary_text(self) -> str:
        n = len(self._known)
        if not n:
            return "No cells yet"
        from collections import Counter
        counts = Counter(self.classes.get(i, self.class_names[0]) for i in self._known)
        detail = ", ".join(f"{v} {name}" for name, v in counts.items())
        text = f"{n} {'cell' if n == 1 else 'cells'}  ·  {detail}"
        report = self._review or {}
        notes = []
        if report.get("holes"):
            notes.append(f"{report['holes']} hole" + ("s" if report["holes"] != 1 else ""))
        if report.get("tiny"):
            notes.append(f"{report['tiny']} tiny")
        if notes:
            text += "  ·  " + ", ".join(notes)
        return text

    def _refresh_buttons(self):
        has = self.labels_layer is not None
        self.glide_btn.setEnabled(has)
        self.points_btn.setEnabled(has)
        self.pan_btn.setEnabled(has)
        busy = self._segment_job is not None and self._segment_job.isRunning()
        self.brush_btn.setEnabled(has)
        self.erase_btn.setEnabled(has)
        self.click_btn.setEnabled(has and not busy)
        self.segment_btn.setEnabled(has and not busy)
        self.segment_choice.setEnabled(has and not busy)
        self.fill_btn.setEnabled(has)
        self.check_btn.setEnabled(has)
        self.fit_btn.setEnabled(has)
        self.contrast_btn.setEnabled(has)
        self.save_btn.setEnabled(has)
        self.package_btn.setEnabled(has)
        self.label_now_btn.setEnabled(has and not busy)
        self.pair_btn.setEnabled(has and bool(self._spot_pairs))
        self.show_infected_btn.setEnabled(has)
        self.show_duplicate_btn.setEnabled(has)
        self.feature_btn.setEnabled(has and not (self._feature_job is not None and self._feature_job.isRunning()))
        self.density_btn.setEnabled(has)
        marked = bool(self._known) or bool(self._rois)
        self.generate_btn.setEnabled(has and marked)
        self.lanes_btn.setEnabled(has)
        self.report_btn.setEnabled(has)
        batch_busy = self._batch_job is not None and self._batch_job.isRunning()
        self.batch_report_btn.setEnabled(has and self.image_path is not None and not batch_busy)
        self.share_btn.setEnabled(has and bool(self._known))
        self.label_studio_btn.setEnabled(has and bool(self._known))
        self.density_3d_btn.setEnabled(has and marked)
        self.density_flat_btn.setEnabled(has)
        images, index = self._image_index()
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index >= 0 and index < len(images) - 1)
        self.undo_btn.setEnabled(has and bool(self.undo_stack))
        self.redo_btn.setEnabled(has and bool(self.redo_stack))
        self._mark_primary(self.open_btn, not has)
        self._mark_primary(self.save_btn, has and not self._segment_draft)
        self._mark_primary(self.label_now_btn, has and not self._known and not busy and not self._segment_draft)
        self._mark_primary(self.generate_btn, has and marked and (self._segment_draft or bool(self._known)))
        self.accept_draft_btn.setVisible(self._segment_draft)
        self.reject_draft_btn.setVisible(self._segment_draft)
        self.summary.setText(self._summary_text())

    # -------------------------------------------------------------------- segment
    def _segment_key(self) -> str:
        key = self.segment_choice.currentData()
        return "classic" if key is None else str(key)

    def _segment_busy(self) -> bool:
        return self._segment_job is not None and self._segment_job.isRunning()

    def segment_image(self) -> bool:
        """Label the whole image on a background thread. Existing outlines are replaced only after a yes."""
        if self.labels_layer is None or self._display_image is None:
            self._set_status("Open an image first.")
            return False
        if self._segment_busy():
            self._set_status("Segmentation is already running.")
            return False
        self.flush()
        if self._known:
            choice = QMessageBox.question(
                self, "Replace outlines?",
                "This image already has outlines. Replace them with a new segmentation? You can undo it.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return False
        reason = self._image_too_large()
        if reason:
            self._set_status(reason)
            return False
        return self._start_segment(None, None)

    def segment_at_point(self, point) -> bool:
        """Label only the object under this click, and leave every other outline alone."""
        if self.labels_layer is None or self._display_image is None:
            self._set_status("Open an image first.")
            return False
        if self._segment_busy():
            self._set_status("Segmentation is already running.")
            return False
        self.flush()
        data = np.asarray(self.labels_layer.data)
        row, col = int(round(float(point[0]))), int(round(float(point[1])))
        if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
            return False
        if int(data[row, col]) > 0:
            self._set_status("That spot is already outlined.")
            return False
        reason = self._image_too_large()
        if reason:
            self._set_status(reason)
            return False
        return self._start_segment((row, col), np.array(data, copy=True))

    def _image_too_large(self) -> str | None:
        height, width = self.labels_layer.data.shape[:2]
        pixels = int(height) * int(width)
        if pixels > MAX_FEATURE_PIXELS:
            return (
                f"This image has {pixels:,} pixels. Segmentation stops at "
                f"{MAX_FEATURE_PIXELS:,} so the computer does not run out of memory."
            )
        return None

    def _start_segment(self, point, labels) -> bool:
        self._segment_token += 1
        token = self._segment_token
        self.segment_btn.setEnabled(False)
        self.click_btn.setEnabled(False)
        self.segment_choice.setEnabled(False)
        self.label_now_btn.setEnabled(False)
        self._backup_timer.stop()
        key = self._segment_key()
        if point is None:
            self._set_status("Segmenting the image… the window stays open while this runs.")
        else:
            self._set_status("Finding the object under the click…")
        # Infected-spot jobs only need brightness (prefetched on open when possible).
        if key == "spots" and point is None:
            if self._gray_cache is not None and tuple(self._gray_cache.shape[:2]) == tuple(
                np.asarray(self._display_image).shape[:2]
            ):
                payload = self._gray_cache
            else:
                from .segment import to_gray

                payload = np.ascontiguousarray(to_gray(self._display_image), dtype=np.float32)
                self._gray_cache = payload
        else:
            payload = np.array(self._display_image, copy=True)
        job = _SegmentJob(payload, key, point, labels)
        job.succeeded.connect(lambda result, t=token: self._on_segment_done(t, result))
        job.failed.connect(lambda message, t=token: self._on_segment_failed(t, message))
        self._segment_job = job
        job.start()
        return True

    def _on_segment_done(self, token: int, result):
        if token != self._segment_token:
            return
        self._segment_job = None
        if not self._backup_timer.isActive():
            self._backup_timer.start()
        self._refresh_buttons()
        if self._closing or self.labels_layer is None:
            return
        if isinstance(result, dict) and "labels" in result:
            self._apply_auto(result["labels"], result.get("classes"), result.get("meta"))
        elif isinstance(result, tuple):
            self._apply_click(result[0], result[1])
        else:
            self._apply_auto(result)

    def _on_segment_failed(self, token: int, message: str):
        if token != self._segment_token:
            return
        self._segment_job = None
        if not self._backup_timer.isActive():
            self._backup_timer.start()
        self._refresh_buttons()
        self._set_status(f"Could not segment: {message}")

    def _apply_auto(self, labels, classes: dict | None = None, meta: dict | None = None):
        labels = io.to_label_dtype(np.asarray(labels))
        if labels.shape != self.labels_layer.data.shape:
            self._set_status("Segmentation did not match this image.")
            return
        ids = tools.label_ids(labels)
        count = int(ids.size)
        if count == 0:
            self._set_status("No objects found.")
            return
        # One undo step for the whole Segment, one paint, one recolor.
        if self._snapshot is not None:
            self._push_undo()
        self._internal = True
        try:
            self.labels_layer.data = labels
        finally:
            self._internal = False
        known = {int(i) for i in ids}
        class_map = dict(self.classes)
        if classes:
            for name in dict.fromkeys(classes.values()):
                if name not in self.class_names:
                    self.class_names.append(name)
            for lab, name in classes.items():
                if int(lab) in known:
                    class_map[int(lab)] = str(name)
        else:
            fresh = self.current_class()
            for lab in known:
                class_map.setdefault(int(lab), fresh)
        for lab in list(class_map):
            if lab not in known:
                class_map.pop(lab, None)
        self.classes = class_map
        self._known = known
        self._snapshot = (np.copy(labels), dict(self.classes))
        self.labels_layer.selected_label = int(tools.next_free_label(labels))
        self.dirty = self._rev != self._saved_rev
        self._spot_pairs = {int(k): int(v) for k, v in ((meta or {}).get("pairs") or {}).items()}
        self._refresh_class_list(self.current_class())
        self._recolor()
        copies = sum(1 for name in (classes or {}).values() if name == "duplicate")
        extra = ""
        if meta and meta.get("capped"):
            from .spots import SPOT_CAP

            extra = f" Showing the {SPOT_CAP} strongest spots."
        if classes:
            self._offer_segment_draft()
            seconds = float((meta or {}).get("seconds") or 0.0)
            timing = f" in {seconds:.2f}s" if seconds > 0 else ""
            if copies:
                self._set_status(
                    f"Draft ready{timing}: {count} infected cells, {copies} duplicates.{extra} "
                    "Keep draft or Reject. Then click Generate."
                )
            else:
                self._set_status(
                    f"Draft ready{timing}: {count} infected cells. None is a duplicate.{extra} "
                    "Keep draft or Reject. Then click Generate."
                )
        else:
            noun = "object" if count == 1 else "objects"
            self._set_status(f"Found {count} {noun}. Correct with the brush. Undo removes this Segment.")
        self._refresh_buttons()
        self._backup()

    def _offer_segment_draft(self):
        self._segment_draft = True
        self.accept_draft_btn.show()
        self.reject_draft_btn.show()
        self._mark_primary(self.generate_btn, True)

    def accept_segment_draft(self):
        self._segment_draft = False
        self.accept_draft_btn.hide()
        self.reject_draft_btn.hide()
        self._set_status(
            f"Kept {len(self._known)} outlines. Click Generate for the densitogram, or edit with the brush."
        )
        self._refresh_buttons()

    def reject_segment_draft(self):
        if not self._segment_draft:
            self._set_status("No draft to reject.")
            return
        self._segment_draft = False
        self.accept_draft_btn.hide()
        self.reject_draft_btn.hide()
        self.undo()
        self._spot_pairs = {}
        self._set_status("Draft rejected. Outlines cleared. Click Label now to try again.")
        self._refresh_buttons()

    def _apply_click(self, mask, box):
        if mask is None or box is None:
            self._set_status("No object there. Click nearer the middle of it.")
            return
        data = np.array(self.labels_layer.data, copy=True)
        crop = np.array(data[box], copy=True)
        mask = np.asarray(mask, bool)
        if mask.shape != crop.shape:
            self._set_status("Could not place that object.")
            return
        free = crop == 0
        if not (mask & free).any():
            self._set_status("That spot is already outlined.")
            return
        new_id = int(tools.next_free_label(data))
        crop[mask & free] = new_id
        data[box] = crop
        self.labels_layer.data = data
        self.flush()
        self._set_status(f"Added 1 object as {self.classes.get(new_id, self.current_class())}. Correct it with the brush.")

    # ------------------------------------------------------------------------- save
    def confirm_discard(self) -> bool:
        self.flush()
        if not self.dirty:
            return True
        choice = QMessageBox.question(
            self, "Save annotations?", "This image has unsaved cells.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save,
        )
        if choice == QMessageBox.Save:
            return self.save()
        if choice == QMessageBox.Discard and self.image_path is not None:
            clear_recovery(self.image_path)
            self.dirty = False
        return choice == QMessageBox.Discard

    def save(self) -> bool:
        if self.labels_layer is None or self.image_path is None:
            self._set_status("Open an image first.")
            return False
        self.flush()
        default = str(self.image_path.with_name(f"{self.image_path.stem}_seg.npy"))
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save annotations", default, SAVE_FILTER, SAVE_KINDS[0][1],
        )
        if not path:
            return False
        return bool(self._write(path, kind_from_filter(chosen)))

    def save_package(self) -> bool:
        """Save outlines and a densitogram CSV beside the image."""
        if self.labels_layer is None or self.image_path is None or self._display_image is None:
            self._set_status("Open an image first.")
            return False
        self.flush()
        if not self._known and not self._rois:
            self._set_status("Label cells or Find lanes first, then save a package.")
            return False
        from .density import analyze_lane, write_density_csv

        extras: list[Path] = []
        if self._known:
            seg_path = self.image_path.with_name(f"{self.image_path.stem}_seg.npy")
            written = self._write(seg_path, "npy")
            if not written:
                return False
            extras.extend(written)
        else:
            written = []
        self._rois = self._rois or self._rois_from_outlines()
        names = []
        results = []
        for item in self._rois:
            names.append(item.get("name") or ("spot" if item.get("kind") == "cell" else "area"))
            results.append(analyze_lane(
                self._display_image,
                roi=item["roi"],
                along=self._density_along,
                dark_bands=self._density_dark,
                notch_percent=float(self._density_notch),
                mask=self._roi_mask(item),
            ))
        if results:
            csv_path = write_density_csv(
                self.image_path.with_name(f"{self.image_path.stem}_density.csv"),
                results,
                names,
            )
            extras.append(csv_path)
        parts = [path.name for path in extras]
        self._set_status(f"Saved package: {', '.join(parts)}." if parts else "Nothing to save.")
        return bool(parts)

    def find_gel_lanes(self) -> bool:
        """Auto-detect blot/gel lanes or band rows and open the densitogram."""
        if self._display_image is None:
            self._set_status("Open an image first.")
            return False
        from .lanes import find_band_rows, find_lanes, lanes_as_rois

        lanes = find_lanes(self._display_image)
        kind = "lane"
        if len(lanes) < 2:
            rows = find_band_rows(self._display_image)
            if rows:
                lanes, kind = rows, "row"
        if not lanes:
            self._set_status("No gel lanes found. Try Find bands, or draw ROIs around each band.")
            return False
        self._rois = lanes_as_rois(lanes, kind=kind)
        self._density_along = "auto"
        self._show_roi_boxes()
        self._refresh_buttons()
        self.open_densitogram()
        self._set_status(f"Found {len(self._rois)} {kind}s. Generate uses Fiji-style optical density.")
        return True

    def find_gel_bands(self) -> bool:
        """Auto-detect individual bands on any gel/blot and open the densitogram."""
        if self._display_image is None:
            self._set_status("Open an image first.")
            return False
        from .lanes import find_band_rows, find_bands, lanes_as_rois

        bands = find_bands(self._display_image)
        rows = find_band_rows(self._display_image)
        if len(bands) >= 3:
            boxes, kind = bands, "band"
        elif len(rows) >= 1:
            boxes, kind = rows, "row"
        else:
            boxes, kind = bands or rows, "band"
        if not boxes:
            self._set_status("No bands found. Draw a box around each band, then click Generate.")
            return False
        self._rois = lanes_as_rois(boxes, kind=kind)
        self._density_along = "auto"
        self._show_roi_boxes()
        self._refresh_buttons()
        self.open_densitogram()
        self._set_status(
            f"Found {len(self._rois)} {kind}s. Peaks use optical density of the band core."
        )
        return True

    def one_click_report(self) -> bool:
        """Write densitogram CSV + profile PNGs beside the image."""
        if self._display_image is None or self.image_path is None:
            self._set_status("Open an image first.")
            return False
        self.flush()
        from .lanes import find_gel_rois, find_lanes, lanes_as_rois
        from .report import densitogram_report

        rois = self._rois_from_outlines() or list(self._rois)
        if not rois:
            rois = lanes_as_rois(find_gel_rois(self._display_image) or find_lanes(self._display_image))
        if not rois:
            self._set_status("Draw spots, Find bands/lanes, or Mark ROI before Report.")
            return False
        self._rois = rois
        out_dir = self.image_path.parent / "ani_reports" / self.image_path.stem
        written = densitogram_report(
            self._display_image,
            out_dir,
            self.image_path.stem,
            rois=rois,
            along=self._density_along,
            dark_bands=self._density_dark,
            notch_percent=float(self._density_notch),
        )
        if not written:
            self._set_status("Could not build the report.")
            return False
        self._set_status(f"Report saved in {out_dir.name}: {len(written)} files (CSV + profiles).")
        return True

    def batch_folder_report(self) -> bool:
        """Densitogram reports for every image in the current folder."""
        if self.image_path is None:
            self._set_status("Open an image from a folder first.")
            return False
        if self._batch_job is not None and self._batch_job.isRunning():
            self._set_status("A batch report is already running.")
            return False
        folder = self.image_path.parent
        out_dir = folder / "ani_reports"
        self.batch_report_btn.setEnabled(False)
        self._set_status(f"Batch report for {folder.name}…")
        job = _BatchReportJob(folder, out_dir, self._density_notch, self._density_dark)
        job.progressed.connect(self._on_batch_progress)
        job.succeeded.connect(self._on_batch_done)
        job.failed.connect(self._on_batch_failed)
        self._batch_job = job
        job.start()
        return True

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._set_status(f"Batch report {done}/{total}: {name}")

    def _on_batch_done(self, written):
        self._batch_job = None
        self._refresh_buttons()
        count = len(written) if written is not None else 0
        self._set_status(f"Batch report done: {count} files in ani_reports.")

    def _on_batch_failed(self, message: str):
        self._batch_job = None
        self._refresh_buttons()
        self._set_status(f"Batch report failed: {message}")

    def share_package_folder(self) -> bool:
        """Copy image + annotations into a shared folder for teammates."""
        if self.labels_layer is None or self.image_path is None:
            self._set_status("Open an image and label cells first.")
            return False
        self.flush()
        if not self._known:
            self._set_status("Label cells first, then Share package.")
            return False
        start = str(self.image_path.parent)
        dest = QFileDialog.getExistingDirectory(self, "Shared folder for Ani packages", start)
        if not dest:
            return False
        from .collab import share_package
        from .report import densitogram_report

        extras: list[Path] = []
        try:
            rois = self._rois_from_outlines()
            if rois:
                report_dir = Path(dest) / "_tmp_report"
                extras = densitogram_report(
                    self._display_image, report_dir, self.image_path.stem, rois=rois,
                    along=self._density_along, dark_bands=self._density_dark,
                    notch_percent=float(self._density_notch),
                )
        except Exception:
            extras = []
        bundle = share_package(
            dest,
            self.image_path,
            np.asarray(self.labels_layer.data),
            list(self.class_names),
            dict(self.classes),
            extras=extras,
        )
        for path in extras:
            try:
                if path.parent.name == "_tmp_report" and path.is_file():
                    # Files were copied into the bundle by share_package; drop the temp folder.
                    pass
            except Exception:
                pass
        try:
            import shutil

            tmp = Path(dest) / "_tmp_report"
            if tmp.is_dir():
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
        self._set_status(f"Shared package ready: {bundle}")
        return True

    def export_label_studio(self) -> bool:
        """Export polygon annotations for Label Studio import."""
        if self.labels_layer is None or self.image_path is None:
            self._set_status("Open an image and label cells first.")
            return False
        self.flush()
        if not self._known:
            self._set_status("Label cells first, then export Label Studio.")
            return False
        default = str(self.image_path.with_name(f"{self.image_path.stem}_label_studio.json"))
        path, _ = QFileDialog.getSaveFileName(self, "Export Label Studio", default, "JSON (*.json)")
        if not path:
            return False
        from .collab import write_label_studio

        written = write_label_studio(
            path,
            np.asarray(self.labels_layer.data),
            list(self.class_names),
            dict(self.classes),
            image_name=self.image_path.name,
        )
        # Also write COCO beside it for tools that prefer that format.
        try:
            write_segmentation(
                Path(path).with_name(f"{self.image_path.stem}_coco.json"),
                np.asarray(self.labels_layer.data),
                list(self.class_names),
                dict(self.classes),
                image_name=self.image_path.name,
                kind="coco",
            )
        except Exception as exc:
            log_error("coco beside label studio", exc)
        self._set_status(f"Exported Label Studio JSON: {written.name}")
        return True

    def extract_features(self) -> bool:
        if self.labels_layer is None or self.image_path is None or self._display_image is None:
            self._set_status("Open an image first.")
            return False
        self.flush()
        if not self._known:
            self._set_status("Mark the cells and set their classes, then extract features.")
            return False
        if self._feature_job is not None and self._feature_job.isRunning():
            self._set_status("Feature extraction is already running.")
            return False
        labels = np.asarray(self.labels_layer.data)
        reason = feature_job_error(int(labels.shape[0]), int(labels.shape[1]), len(self._known))
        if reason:
            self._set_status(reason)
            return False
        default = str(self.image_path.parent)
        path = QFileDialog.getExistingDirectory(self, "Choose a folder for cell features", default)
        if not path:
            return False
        self.feature_btn.setEnabled(False)
        self._set_status(f"Measuring {len(self._known)} cells…")
        payload = {
            "out_dir": Path(path) / f"{self.image_path.stem}_features",
            "image": np.array(self._display_image, copy=True),
            "labels": np.array(labels, copy=True),
            "classes": dict(self.classes),
            "image_name": self.image_path.name,
            "well": self.well_edit.text().strip(),
            "slide": self.slide_edit.text().strip(),
            "magnification": self.mag_edit.text().strip(),
            "pixel_size_um": self._pixel_size(),
        }
        job = _FeatureJob(payload)
        job.progressed.connect(self._on_feature_progress)
        job.succeeded.connect(self._on_feature_done)
        job.failed.connect(self._on_feature_failed)
        self._feature_job = job
        job.start()
        return True

    def _on_feature_progress(self, done: int, total: int):
        if total:
            self._set_status(f"Measuring cell {done} of {total}…")

    def _on_feature_done(self, path: str):
        self._feature_job = None
        self._refresh_buttons()
        folder = Path(path).parent
        self._set_status(f"Saved features, patches, masks and a QC image to {folder.name}.")

    def _on_feature_failed(self, message: str):
        self._feature_job = None
        self._refresh_buttons()
        self._set_status(f"Could not extract features: {message}")

    def extract_to(self, path) -> list[Path]:
        """Measure every marked cell and write the training folder. Returns the CSV."""
        if self.labels_layer is None or self._display_image is None:
            raise RuntimeError("Open an image first.")
        self.flush()
        if not self._known:
            raise RuntimeError("Mark the cells and set their classes, then extract features.")
        written = export_training_set(
            path, self._display_image, np.asarray(self.labels_layer.data), dict(self.classes),
            image_name=self.image_path.name if self.image_path else "",
            well=self.well_edit.text().strip(), slide=self.slide_edit.text().strip(),
            magnification=self.mag_edit.text().strip(),
            pixel_size_um=self._pixel_size(),
        )
        self._set_status(f"Saved features, patches, masks and a QC image to {written.parent.name}.")
        return [written]

    def save_to(self, path) -> list[Path]:
        """Write annotations without a dialog. Returns the files that were written."""
        if self.labels_layer is None:
            raise RuntimeError("Open an image first.")
        self.flush()
        written = self._write(path, "all")
        if not written:
            raise RuntimeError(self.status.text())
        return written

    def _backup(self):
        """Copy unsaved outlines aside so a crash or power cut can restore them."""
        if not self.dirty or self.labels_layer is None or self.image_path is None:
            return
        try:
            save_recovery(
                self.image_path, np.asarray(self.labels_layer.data),
                list(self.class_names), dict(self.classes),
            )
        except Exception as exc:
            log_error("recovery copy", exc)

    def _undo_bytes(self) -> int:
        return sum(int(item[0].nbytes) for item in self.undo_stack)

    def _write(self, path, kind: str = "all") -> list[Path]:
        try:
            info = self.display_info or {}
            written = write_segmentation(
                Path(path), np.asarray(self.labels_layer.data), list(self.class_names), dict(self.classes),
                image_name=self.image_path.name if self.image_path else "",
                plane=info.get("plane"), channel=info.get("channel"), kind=kind,
            )
        except Exception as e:
            log_error("save annotations", e)
            self._set_status(f"Could not save: {e}")
            return []
        self._saved_rev = self._rev
        self.dirty = False
        if self.image_path is not None:
            clear_recovery(self.image_path)
        names = ", ".join(p.name for p in written)
        self._set_status(f"Saved {len(self._known)} cells. {names}")
        self.status.setToolTip("\n".join(str(p) for p in written))
        return written
