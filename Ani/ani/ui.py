"""Ani main panel, docked inside a napari viewer.

Tabs: Annotate (correct labels) · AI (auto-label, detector, review queue) · Train (evaluate, improve, export)
· Settings (language, help). Slow work runs in background threads so the window never freezes.
"""
from __future__ import annotations

import inspect
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__, ai, batch, io, tools
from .project import STATUSES, Project

STATUS_ICON = {"todo": "○", "in progress": "✎", "flagged": "⚑", "done": "✔"}
STATUS_TEXT = {"todo": "to do", "in progress": "in progress", "flagged": "flagged", "done": "done"}
CLASS_COLOURS = ["#ff4d4d", "#4dd2ff", "#ffd24d", "#7dff4d", "#d24dff", "#ff9f4d", "#4dffb8", "#ff4da6"]
UNDO_BUDGET_BYTES = 400 * 1024 * 1024   # keep undo memory small for 8 GB laptops
BACKUP_EVERY_MS = 60_000                # power-cut protection
PAINT_SETTLE_MS = 300                   # one brush stroke = one undo step

# tool key -> (napari labels mode or None for Ani click tools, shortcut, tooltip)
TOOLS = {
    "Inspect": ("pan_zoom", "Shift-I", "Pan/zoom. Click an object to see its number, size and class."),
    "Paint": ("paint", "Shift-P", "Paint with the current object number (brush size in the layer controls)."),
    "Erase": ("erase", "Shift-E", "Erase pixels."),
    "Fill": ("fill", "Shift-F", "Flood-fill an area with the current object number."),
    "New cell": ("paint", "Shift-N", "Paint a brand-new object (one the detector MISSED)."),
    "Outline": ("polygon", "Shift-O", "Draw the outline of a MISSED object: click around it, double-click to close."),
    "AI click": (None, "Shift-A", "Click inside a MISSED object: the AI detector outlines it for you."),
    "Delete": (None, "Shift-D", "Click an object to delete it (FALSE detection)."),
    "Merge": (None, "Shift-M", "Click two objects: the second is merged into the first."),
    "Split": (None, "Shift-X", "Click an object to split it into N parts (two cells detected as one)."),
    "Set class": (None, "Shift-C", "Click an object to give it the selected class."),
}
CLICK_TOOLS = ("Inspect", "Delete", "Merge", "Split", "Set class", "AI click")


class _CloseFilter(QObject):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        from qtpy.QtCore import QEvent
        if event.type() == QEvent.Close:
            self._callback()
        return False


class _Bridge(QObject):
    """Carries results from the save thread back to the UI thread (Qt queues the signal)."""
    done = Signal(object)


def _slim_history(layer):
    """Ani has its own undo; napari's per-stroke history (100 steps by default) would only waste memory."""
    try:
        layer._history_limit = 1
        layer._reset_history()
    except Exception:
        pass


def _guard(fn):
    """Show errors in the panel instead of crashing or failing silently."""
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as e:  # shown to the user
            import traceback
            traceback.print_exc()
            self.say(f"⚠ {type(e).__name__}: {e}")
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class WrapCheck(QWidget):
    """A checkbox whose label wraps (long translated texts must not widen the panel)."""

    def __init__(self, text: str):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.box = QCheckBox()
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.mousePressEvent = lambda e: self.box.toggle()
        row.addWidget(self.box, 0, Qt.AlignTop)
        row.addWidget(self.label, 1)
        self.toggled = self.box.toggled
        self.isChecked = self.box.isChecked
        self.setChecked = self.box.setChecked

    def setToolTip(self, tip):  # noqa: N802 (Qt naming)
        self.box.setToolTip(tip)
        self.label.setToolTip(tip)


def _wrap_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    return lab


def _button(text, fn, tip=""):
    b = QPushButton(text)
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(lambda *_: fn())
    return b


class AniWidget(QWidget):
    def __init__(self, viewer, project_dir: str | None = None):
        super().__init__()
        self.viewer = viewer
        self.project: Project | None = None
        self.stem: str | None = None
        self.label_set: str | None = None
        self.image_raw = None
        self.layout_info = None
        self.image_shape = None
        self.search_image = None            # single-channel image used by AI and the problem finder
        self.classes: dict[int, str] = {}
        self.labels_layer = None
        self.pred_layer = None
        self.problem_layer = None
        self.problems: list[dict] = []
        self.problem_i = -1
        self._current: tuple[np.ndarray, dict] | None = None   # last committed state (private copy)
        self.undo_stack: list[tuple[np.ndarray, dict]] = []
        self.redo_stack: list[tuple[np.ndarray, dict]] = []
        self._paint_pending = False
        self._busy = False
        self.dirty = False
        self.tool = "Inspect"
        self.merge_first: int | None = None
        self.backends: list = ai.backends_for(None)
        self.jobs: dict[str, object] = {}
        self.sync_jobs = False               # tests run background jobs synchronously
        self.ask = self._ask_dialog          # replaced in tests
        self._tune_result = None
        self._image_layers: list = []
        self._image_key = None
        self._saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ani-save")  # ordered disk writes
        self._pending_saves: set[tuple[str, str]] = set()
        self._bridge = _Bridge()
        self._bridge.done.connect(self._save_finished)

        self._build_ui()
        self._bind_keys()
        self._paint_timer = QTimer(self)
        self._paint_timer.setSingleShot(True)
        self._paint_timer.setInterval(PAINT_SETTLE_MS)
        self._paint_timer.timeout.connect(self._flush_paint)
        self._backup_timer = QTimer(self)
        self._backup_timer.setInterval(BACKUP_EVERY_MS)
        self._backup_timer.timeout.connect(self.backup_now)
        self._backup_timer.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_quit)
        try:  # closing the viewer window: finish saves and stop threads before Qt deletes objects
            self._close_filter = _CloseFilter(self._on_quit)
            viewer.window._qt_window.installEventFilter(self._close_filter)
        except Exception:
            pass
        if project_dir:
            self.open_project(project_dir)

    # ================================================================== UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(QLabel(f"<b style='font-size:15px'>Ani</b> &nbsp;"
                               f"<span style='color:#999'>{'cell annotator'} v{__version__}</span>"))
        self.tabs = QTabWidget()
        self._pages = []
        outer.addWidget(self.tabs, 1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setMinimumHeight(36)
        outer.addWidget(self.info)
        for title, builder in (("Annotate", self._build_annotate), ("AI", self._build_ai),
                               ("Train", self._build_train), ("Settings", self._build_settings)):
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(4, 4, 4, 4)
            builder(lay)
            lay.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.tabs.addTab(scroll, title)
            self._pages.append((scroll, page))
        self.setMinimumWidth(390)
        # drop-downs may shrink to the panel width (their list still shows full text) so long
        # translated entries never push the panel wider than the screen
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(6)
            combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 24)

    def _fit_width(self):
        """Every tab keeps room for its vertical scrollbar so nothing is cut off on the right.
        Measured once styled (sizes before the first show are too small)."""
        need = max(page.minimumSizeHint().width() for _, page in self._pages)
        for scroll, _ in self._pages:
            scroll.setMinimumWidth(need + scroll.verticalScrollBar().sizeHint().width() + 6)
        self.setMinimumWidth(max(390, self.minimumSizeHint().width()))   # lets napari's dock grow to fit

    def showEvent(self, event):  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._fit_width()
        QTimer.singleShot(0, self._fit_width)

    # ---------------------------------------------------------------- Annotate tab
    def _build_annotate(self, lay):
        box = QGroupBox("1. Project")
        g = QGridLayout(box)
        self.project_label = QLabel("No project open")
        self.project_label.setWordWrap(True)
        g.addWidget(self.project_label, 0, 0, 1, 3)
        g.addWidget(_button("Open / create…", self._choose_project,
                            "Any folder. Ani creates images/, predictions/, labels/ inside."), 1, 0)
        g.addWidget(_button("Add images…", self._import_images, "Copy images into this project."), 1, 1)
        g.addWidget(_button("Demo project", self._make_demo,
                            "Practice project: synthetic cells + a faulty detector output."), 1, 2)
        lay.addWidget(box)

        box = QGroupBox("2. Images")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("all", "all")
        for st in STATUSES:
            self.filter_combo.addItem(STATUS_TEXT[st], st)
        self.filter_combo.currentIndexChanged.connect(lambda _: self.refresh_list())
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("by name", "name")
        self.sort_combo.addItem("worst first", "worst")
        self.sort_combo.setToolTip(("Worst first: images with the most suspected mistakes at the top "
                                      "(press 'Rank all images' in the AI tab)."))
        self.sort_combo.currentIndexChanged.connect(lambda _: self.refresh_list())
        self.axis_combo = QComboBox()
        for key, text in (("auto", "auto"), ("channels", "channels"), ("z", "z")):
            self.axis_combo.addItem(text, key)
        self.axis_combo.setToolTip("For 3-D arrays: is the first axis colour channels or Z-slices?")
        self.axis_combo.currentIndexChanged.connect(lambda _: self.stem and self.load_image(self.stem))
        row.addWidget(self.filter_combo, 1)
        row.addWidget(self.sort_combo, 1)
        row.addWidget(QLabel("Axis 0"))
        row.addWidget(self.axis_combo)
        v.addLayout(row)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(100)
        self.image_list.itemClicked.connect(lambda it: self.load_image(it.data(Qt.UserRole)))
        v.addWidget(self.image_list)
        row = QHBoxLayout()
        row.addWidget(_button("◀ " + "Prev" + "  (PgUp)", lambda: self.step_image(-1)))
        row.addWidget(_button("Next" + " ▶  (PgDn)", lambda: self.step_image(1)))
        v.addLayout(row)
        self.show_pred = WrapCheck("Show detector output as outline")
        self.show_pred.toggled.connect(self._toggle_prediction)
        v.addWidget(self.show_pred)
        lay.addWidget(box)

        box = QGroupBox("3. What to label")
        g = QGridLayout(box)
        g.addWidget(QLabel("Label set"), 0, 0)
        self.set_combo = QComboBox()
        self.set_combo.setToolTip(("Objects that may overlap go in different sets: cells, nuclei, nucleoli, "
                                     "mitochondria…"))
        self.set_combo.currentTextChanged.connect(self._on_set_changed)
        g.addWidget(self.set_combo, 0, 1)
        g.addWidget(_button("+ " + "set", self._add_set), 0, 2)
        g.addWidget(QLabel("Class"), 1, 0)
        self.class_combo = QComboBox()
        g.addWidget(self.class_combo, 1, 1)
        g.addWidget(_button("+ " + "class", self._add_class), 1, 2)
        self.colour_by_class = WrapCheck("Colour by class")
        self.colour_by_class.toggled.connect(lambda _: self._update_colours())
        self.outlines_only = WrapCheck("Outlines only")
        self.outlines_only.toggled.connect(self._apply_outline_view)
        self.show_other_sets = WrapCheck("Show other sets")
        self.show_other_sets.setChecked(True)
        self.show_other_sets.toggled.connect(self._toggle_other_sets)
        g.addWidget(self.colour_by_class, 2, 0, 1, 2)
        g.addWidget(self.outlines_only, 2, 2)
        g.addWidget(self.show_other_sets, 3, 0, 1, 2)
        lay.addWidget(box)

        box = QGroupBox("4. Correct")
        g = QGridLayout(box)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for i, (key, (_, shortcut, tip)) in enumerate(TOOLS.items()):
            b = QPushButton(key)
            b.setCheckable(True)
            b.setProperty("tool", key)
            b.setToolTip(f"{tip}\n{'Shortcut'}: {shortcut}")
            b.setChecked(key == "Inspect")
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addButton(b)
            g.addWidget(b, i // 3, i % 3)
        r = (len(TOOLS) + 2) // 3
        g.addWidget(QLabel("Split into"), r, 0)
        self.split_parts = QSpinBox()
        self.split_parts.setRange(2, 8)
        g.addWidget(self.split_parts, r, 1)
        r += 1
        g.addWidget(_button("↶ " + "Undo", self.undo,
                            "Undo any change: tools, brush strokes, classes." + "  Ctrl+Z / Shift-U"), r, 0)
        g.addWidget(_button("↷ " + "Redo", self.redo, "Ctrl+Shift+Z / Shift-Y"), r, 1)
        r += 1
        self.min_size = QSpinBox()
        self.min_size.setRange(1, 100000)
        self.min_size.setValue(30)
        self.min_size.setSuffix(" px")
        g.addWidget(_button("Remove smaller than",
                            lambda: self.apply(tools.remove_small, self.min_size.value(),
                                               msg="Removed tiny objects.")), r, 0, 1, 2)
        g.addWidget(self.min_size, r, 2)
        r += 1
        g.addWidget(_button("Fill holes", lambda: self.apply(tools.fill_holes, msg="Filled holes.")), r, 0)
        g.addWidget(_button("Renumber", self._renumber, "Renumber objects 1..N without gaps"), r, 1)
        g.addWidget(_button("Clear all", self._clear_all), r, 2)
        lay.addWidget(box)

        box = QGroupBox("5. Find problems")
        g = QGridLayout(box)
        g.addWidget(_button("🔍 " + "Find problems" + "  (Shift-R)", self.find_problems,
                            ("Lists objects that look wrong (fragments, giants, merged shapes, broken pieces) "
                               "and visible objects nobody labelled (missed cells).")), 0, 0, 1, 2)
        self.search_missed = WrapCheck("missed too")
        self.search_missed.setToolTip("Also look for visible objects that have no label (missed cells).")
        self.search_missed.setChecked(True)
        g.addWidget(self.search_missed, 0, 2)
        g.addWidget(_button("◀  (Shift-J)", lambda: self.goto_problem(-1)), 1, 0)
        g.addWidget(_button("▶  (Shift-K)", lambda: self.goto_problem(1)), 1, 1)
        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        g.addWidget(self.problem_label, 2, 0, 1, 3)
        lay.addWidget(box)

        box = QGroupBox("6. Save")
        g = QGridLayout(box)
        g.addWidget(_button("💾 " + "Save" + "  (Shift-S)", self.save,
                            "Save corrections for this image and label set."), 0, 0)
        g.addWidget(_button("✔ " + "Save + Done → next", self.save_done_next,
                            "Save, mark this image finished, open the next one."), 0, 1)
        g.addWidget(_button("⚑ " + "Flag", lambda: self._set_status("flagged"),
                            "Hard case / unsure: review later."), 1, 0)
        g.addWidget(_button("↺ " + "Reload", self._reload, "Discard unsaved changes and reload from disk."),
                    1, 1)
        self.autosave = WrapCheck("Auto-save when changing image")
        self.autosave.setChecked(True)
        g.addWidget(self.autosave, 2, 0, 1, 2)
        lay.addWidget(box)

    # ---------------------------------------------------------------- AI tab
    def _build_ai(self, lay):
        box = QGroupBox("Detector")
        g = QGridLayout(box)
        self.backend_combo = QComboBox()
        self.backend_combo.currentIndexChanged.connect(lambda _: self._backend_changed())
        g.addWidget(self.backend_combo, 0, 0, 1, 2)
        g.addWidget(_button("Refresh", self.refresh_backends,
                            "Look again for installed AI packages and your trained models."), 0, 2)
        self.backend_status = QLabel("")
        self.backend_status.setWordWrap(True)
        g.addWidget(self.backend_status, 1, 0, 1, 3)
        g.addWidget(QLabel("Cell diameter"), 2, 0)
        self.diameter = QSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setSpecialValueText("auto")
        self.diameter.setSuffix(" px")
        self.diameter.setToolTip("0 = estimate automatically")
        g.addWidget(self.diameter, 2, 1)
        g.addWidget(QLabel("Objects are"), 4, 0)
        self.polarity = QComboBox()
        self.polarity.addItem("auto (learn from labelled cells)", "auto")
        self.polarity.addItem("brighter than background (fluorescence)", "bright")
        self.polarity.addItem("darker than background (brightfield)", "dark")
        g.addWidget(self.polarity, 4, 1, 1, 2)
        g.addWidget(QLabel("Channel"), 3, 0)
        self.channel = QSpinBox()
        self.channel.setRange(0, 9)
        self.channel.setToolTip("Which channel of a multi-channel image the AI should look at.")
        self.channel.valueChanged.connect(lambda _: self._update_search_image())
        g.addWidget(self.channel, 3, 1)
        lay.addWidget(box)

        box = QGroupBox("This image")
        g = QGridLayout(box)
        g.addWidget(_button("Auto-label (replace)", lambda: self.auto_label("replace"),
                            "Throw away the current objects and let the detector label everything."), 0, 0)
        g.addWidget(_button("Auto-label (add missing)", lambda: self.auto_label("add"),
                            "Keep your objects; add only new ones the detector finds."), 0, 1)
        g.addWidget(_wrap_label("Tip: the 'AI click' tool (Shift-A) outlines one missed cell where you click."),
                    1, 0, 1, 2)
        lay.addWidget(box)

        box = QGroupBox("All images")
        g = QGridLayout(box)
        self.use_command = WrapCheck("Use my own detector program")
        self.use_command.setToolTip("Your program is run once per image. Use {input} and {output} in the command.")
        g.addWidget(self.use_command, 0, 0, 1, 3)
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("python my_detector.py --input {input} --output {output}")
        self.command_edit.editingFinished.connect(self._save_command)
        g.addWidget(self.command_edit, 1, 0, 1, 3)
        self.overwrite = WrapCheck("Redo images that already have detector output")
        g.addWidget(self.overwrite, 2, 0, 1, 3)
        g.addWidget(_button("▶ " + "Run detector on all images", self.run_batch), 3, 0, 1, 2)
        self.stop_batch = _button("■ " + "Stop", lambda: self.stop_job("batch"))
        self.stop_batch.setEnabled(False)
        g.addWidget(self.stop_batch, 3, 2)
        g.addWidget(_button("⚠ " + "Rank all images (worst first)", self.rank_all,
                            "Count suspected mistakes on every image so you can fix the worst first."),
                    4, 0, 1, 3)
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        g.addWidget(self.batch_progress, 5, 0, 1, 3)
        self.batch_log = QLabel("")
        self.batch_log.setWordWrap(True)
        g.addWidget(self.batch_log, 6, 0, 1, 3)
        lay.addWidget(box)

    # ---------------------------------------------------------------- Train tab
    def _build_train(self, lay):
        box = QGroupBox("1. How good is the detector?")
        g = QGridLayout(box)
        self.eval_source = QComboBox()
        self.eval_source.addItem("Saved detector output", "saved")
        self.eval_source.addItem("Selected AI detector (runs it now)", "backend")
        g.addWidget(self.eval_source, 0, 0, 1, 2)
        g.addWidget(_button("Evaluate", self.evaluate,
                            "Compare with your corrections on images marked Done."), 0, 2)
        self.eval_table = QTableWidget(0, 7)
        self.eval_table.setHorizontalHeaderLabels(["When", "Detector", "Images", "Precision",
                                                   "Recall", "F1", "IoU"])
        self.eval_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.eval_table.setMinimumHeight(130)
        self.eval_table.verticalHeader().setVisible(False)
        g.addWidget(self.eval_table, 1, 0, 1, 3)
        expl = QLabel(("Precision: how many found objects are real. Recall: how many real objects were found. "
                         "F1 combines both (1.0 = perfect)."))
        expl.setWordWrap(True)
        expl.setStyleSheet("color:#999")
        g.addWidget(expl, 2, 0, 1, 3)
        lay.addWidget(box)

        box = QGroupBox("2. Improve it")
        g = QGridLayout(box)
        g.addWidget(QLabel("Trials"), 0, 0)
        self.tune_trials = QSpinBox()
        self.tune_trials.setRange(4, 200)
        self.tune_trials.setValue(30)
        g.addWidget(self.tune_trials, 0, 1)
        g.addWidget(_button("Tune Classic", self.tune,
                            "The built-in detector learns its settings from your corrections (CPU, minutes)."),
                    0, 2)
        g.addWidget(QLabel("Epochs"), 1, 0)
        self.ft_epochs = QSpinBox()
        self.ft_epochs.setRange(5, 2000)
        self.ft_epochs.setValue(100)
        g.addWidget(self.ft_epochs, 1, 1)
        self.ft_base = QComboBox()
        self.ft_base.addItems(["cyto3", "nuclei"])
        g.addWidget(self.ft_base, 2, 1)
        g.addWidget(QLabel("Start from"), 2, 0)
        self.ft_button = _button("Fine-tune Cellpose", self.finetune,
                                 ("Re-train Cellpose on your corrections. Slow on CPU – or use the Colab "
                                    "notebook from the export for a free GPU."))
        g.addWidget(self.ft_button, 1, 2, 2, 1)
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        g.addWidget(self.train_progress, 3, 0, 1, 3)
        self.train_log = QLabel("")
        self.train_log.setWordWrap(True)
        g.addWidget(self.train_log, 4, 0, 1, 3)
        lay.addWidget(box)

        box = QGroupBox("3. Export training data")
        g = QGridLayout(box)
        self.fmt = {k: QCheckBox(k) for k in ("masks", "coco", "yolo")}
        for i, cb in enumerate(self.fmt.values()):
            cb.setChecked(True)
            g.addWidget(cb, 0, i)
        self.only_done = WrapCheck("Only images marked Done")
        self.only_done.setChecked(True)
        g.addWidget(self.only_done, 1, 0, 1, 3)
        self.to_png = WrapCheck("8-bit PNG images for COCO/YOLO")
        self.to_png.setToolTip("Most detection trainers can't read 16-bit or multi-channel TIFF.")
        self.to_png.setChecked(True)
        g.addWidget(self.to_png, 2, 0, 1, 3)
        self.with_notebooks = WrapCheck("Add Google Colab notebooks (free GPU training)")
        self.with_notebooks.setChecked(True)
        g.addWidget(self.with_notebooks, 3, 0, 1, 3)
        g.addWidget(QLabel("Validation"), 4, 0)
        self.val_pct = QSpinBox()
        self.val_pct.setRange(0, 50)
        self.val_pct.setValue(20)
        self.val_pct.setSuffix("%")
        g.addWidget(self.val_pct, 4, 1)
        g.addWidget(_button("Export dataset", self._export), 5, 0, 1, 3)
        lay.addWidget(box)

    # ---------------------------------------------------------------- Settings tab
    def _build_settings(self, lay):
        box = QGroupBox("AI packages on this computer")
        v = QVBoxLayout(box)
        self.pkg_label = QLabel("")
        self.pkg_label.setWordWrap(True)
        self.pkg_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.pkg_label)
        lay.addWidget(box)

        box = QGroupBox("Keyboard shortcuts")
        v = QVBoxLayout(box)
        rows = [(s, k) for k, (_, s, _) in TOOLS.items()] + [
            ("Ctrl+Z / Shift-U", "Undo"), ("Ctrl+Shift+Z / Shift-Y", "Redo"),
            ("Shift-S", "Save"), ("Shift-R", "Find problems"),
            ("Shift-K / Shift-J", "Next / previous problem"), ("PgDn / PgUp", "Next / previous image")]
        v.addWidget(QLabel("<table>" + "".join(f"<tr><td><b>{k}</b>&nbsp;&nbsp;</td><td>{t}</td></tr>"
                                                for k, t in rows) + "</table>"))
        lay.addWidget(box)

        box = QGroupBox("About")
        v = QVBoxLayout(box)
        about = QLabel(("Ani is free software (MIT licence) for correcting cell segmentations and building "
                          "training data on ordinary computers. Works offline."))
        about.setWordWrap(True)
        v.addWidget(about)
        lay.addWidget(box)
        self._refresh_packages()

    def _refresh_packages(self):
        lines = []
        for b in self.backends:
            ok, msg = b.available()
            lines.append(f"{'✔' if ok else '✗'} {b.name}: {msg if ok else msg}")
        self.pkg_label.setText("\n".join(lines))

    def _bind_keys(self):
        actions = {
            "Shift-S": self.save, "Shift-U": self.undo, "Shift-Y": self.redo,
            "Shift-R": self.find_problems, "Shift-K": lambda: self.goto_problem(1),
            "Shift-J": lambda: self.goto_problem(-1),
            "PageDown": lambda: self.step_image(1), "PageUp": lambda: self.step_image(-1),
        }
        for key, (_, shortcut, _) in TOOLS.items():
            actions[shortcut] = (lambda k=key: self.set_tool(k))
        for shortcut, fn in actions.items():
            self.viewer.bind_key(shortcut, lambda v, f=fn: f(), overwrite=True)

    def say(self, msg: str):
        self.viewer.status = msg
        self.info.setText(msg)

    def _ask_dialog(self, question: str) -> bool:
        return QMessageBox.question(self, "Ani", question) == QMessageBox.Yes

    # ================================================================== background jobs
    def start_job(self, kind: str, fn, *args, on_yield=None, on_done=None, on_error=None, **kwargs) -> bool:
        """Run fn(*args) off the UI thread. Generators report progress through on_yield."""
        if kind in self.jobs:
            self.say("Please wait – this is still running.")
            return False
        on_error = on_error or (lambda e: self.say("⚠ " + str(e)))
        on_yield = self._safe(on_yield) if on_yield else None
        on_done = self._safe(on_done) if on_done else None
        if self.sync_jobs:
            try:
                res = fn(*args, **kwargs)
                if inspect.isgenerator(res):
                    for item in res:
                        if on_yield:
                            on_yield(item)
                    res = None
                if on_done:
                    on_done(res)
            except Exception as e:
                on_error(e)
            self._job_finished(kind)
            return True
        from napari.qt.threading import create_worker

        worker = create_worker(fn, *args, _start_thread=False, **kwargs)
        if on_yield is not None and hasattr(worker, "yielded"):
            worker.yielded.connect(on_yield)
        if on_done is not None:
            worker.returned.connect(on_done)
        worker.errored.connect(on_error)
        worker.finished.connect(lambda k=kind: self._job_finished(k))
        self.jobs[kind] = worker
        self._job_started(kind)
        worker.start()
        return True

    def _safe(self, fn):
        """Callbacks from background jobs run inside Qt signals: show their errors instead of losing them."""
        def wrapper(*a):
            try:
                return fn(*a)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.say(f"⚠ {type(e).__name__}: {e}")
        return wrapper

    def stop_job(self, kind: str):
        w = self.jobs.get(kind)
        if w is not None and hasattr(w, "quit"):
            w.quit()
            self.say("Stopping after the current image…")

    def _job_started(self, kind):
        if kind == "batch":
            self.stop_batch.setEnabled(True)
            self.batch_progress.setVisible(True)
        if kind == "train":
            self.train_progress.setVisible(True)

    def _job_finished(self, kind):
        self.jobs.pop(kind, None)
        if kind == "batch":
            self.stop_batch.setEnabled(False)
            self.batch_progress.setVisible(False)
        if kind == "train":
            self.train_progress.setVisible(False)

    # ================================================================== project
    def _choose_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose or create an Ani project folder")
        if d:
            self.open_project(d)

    def _make_demo(self):
        d = QFileDialog.getExistingDirectory(self, "Where should the demo project be created?")
        if not d:
            return
        from .demo import make_demo_project

        root = Path(d) / "Ani-demo"
        if not (root / "project.json").exists():
            make_demo_project(root)
        self.open_project(root)
        self.say(("Demo ready. Each image has missed cells, false detections, merged and split cells. "
                    "Press 'Find problems' to locate them."))

    @_guard
    def open_project(self, root):
        if not self._confirm_unsaved():
            return
        self.project = Project(Path(root))
        self.stem = None
        self.label_set = self.project.label_sets[0]
        self.project_label.setText(f"<b>{self.project.root.name}</b><br>"
                                   f"<span style='color:#999'>{self.project.root}</span>")
        self.command_edit.setText(self.project.meta.get("detector_command", ""))
        self.refresh_backends()
        self.refresh_sets()
        self.refresh_list()
        self._refresh_eval_table()
        imgs = self.project.images()
        if not imgs:
            self.say("Project ready. Put images into:" + f"\n{self.project.images_dir}\n" +
                     "and your detector's masks (same file name) into:" + f"\n{self.project.predictions_dir}")
            return
        first = next((p for p in imgs if self.project.status(p.stem) in ("todo", "in progress")), imgs[0])
        self.load_image(first.stem, check_unsaved=False)

    def _import_images(self):
        if not self.project:
            return self.say("Open a project first.")
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "",
                                                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        n = self.project.import_images(files)
        self.refresh_list()
        self.say("Added {n} image(s).".format(n=n))

    def refresh_list(self):
        self.image_list.clear()
        if not self.project:
            return
        want = self.filter_combo.currentData()
        imgs = self.project.images()
        if self.sort_combo.currentData() == "worst":
            imgs = sorted(imgs, key=lambda p: -batch.problem_value(self.project, p.stem, self.label_set))
        for p in imgs:
            if want != "all" and self.project.status(p.stem) != want:
                continue
            text, tip = self._item_text(p.stem)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, p.stem)
            item.setToolTip(tip)
            self.image_list.addItem(item)
            if p.stem == self.stem:
                self.image_list.setCurrentItem(item)

    def _item_text(self, stem: str):
        st = self.project.status(stem)
        path = self.project.image_path(stem)
        name = path.name if path else stem
        score = batch.problem_value(self.project, stem, self.label_set)
        text = f"{STATUS_ICON[st]}  {name}" + (f"   ⚠ {score:.0f}" if score > 0 else "")
        stats = self.project.meta["images"].get(stem, {}).get("stats", {})
        tip = [STATUS_TEXT[st]] + [f"{s}: {d['cells']} objects – detector missed {d['added']}, wrongly found "
                                   f"{d['removed']}, reshaped {d['modified']}" for s, d in stats.items()]
        return text, "\n".join(tip)

    def _update_list_item(self, stem: str, select: bool = False):
        """Update one row instead of rebuilding the whole list (fast with thousands of images)."""
        if not self.project:
            return
        want = self.filter_combo.currentData()
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == stem:
                if want != "all" and self.project.status(stem) != want and stem != self.stem:
                    self.image_list.takeItem(i)
                    return
                text, tip = self._item_text(stem)
                item.setText(text)
                item.setToolTip(tip)
                if select:
                    self.image_list.setCurrentItem(item)
                return

    def refresh_sets(self):
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        if self.project:
            self.set_combo.addItems(self.project.label_sets)
            self.set_combo.setCurrentText(self.label_set)
        self.set_combo.blockSignals(False)
        self.refresh_classes()

    def refresh_classes(self):
        cur = self.class_combo.currentText()
        self.class_combo.clear()
        if self.project and self.label_set:
            names = self.project.classes(self.label_set)
            self.class_combo.addItems(names)
            if cur in names:
                self.class_combo.setCurrentText(cur)

    def refresh_backends(self):
        keep = self.backend_combo.currentData()
        self.backends = ai.backends_for(self.project)
        self.backend_combo.blockSignals(True)
        self.backend_combo.clear()
        for b in self.backends:
            ok, _ = b.available()
            self.backend_combo.addItem(("✔ " if ok else "✗ ") + b.name, b.key)
        idx = self.backend_combo.findData(keep) if keep else -1
        self.backend_combo.setCurrentIndex(max(idx, 0))
        self.backend_combo.blockSignals(False)
        self._backend_changed()
        self._refresh_packages()
        try:
            from .ai.cellpose_backend import CellposeBackend

            ok, msg = CellposeBackend().available()
            self.ft_button.setEnabled(ok)
            if not ok:
                self.ft_button.setToolTip(f"Cellpose: {msg}")
        except Exception:
            self.ft_button.setEnabled(False)

    def backend(self):
        i = self.backend_combo.currentIndex()
        return self.backends[i] if 0 <= i < len(self.backends) else self.backends[0]

    def _backend_changed(self):
        b = self.backend()
        ok, msg = b.available()
        what = []
        if b.can_auto:
            what.append("labels whole images")
        if b.can_click:
            what.append("AI click")
        self.backend_status.setText(("✔ " + "ready" if ok else "✗ " + msg) + " · " + ", ".join(what))

    def _save_command(self):
        if self.project:
            self.project.set_meta("detector_command", self.command_edit.text().strip())

    def _add_set(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "New label set",
                                        ("Name of objects that can overlap the others\n"
                                           "(e.g. nuclei, nucleoli, mitochondria):"))
        if ok and name.strip():
            key = self.project.add_label_set(name)
            self.refresh_sets()
            self.set_combo.setCurrentText(key)

    def _add_class(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "New class",
                                        "New class inside '{set}'\n(e.g. dividing cell, apoptotic, stained):"
                                        .format(set=self.label_set))
        if ok and name.strip():
            self.project.add_class(self.label_set, name)
            self.refresh_classes()
            self.class_combo.setCurrentText(name.strip())
            self._update_colours()

    def _on_set_changed(self, name: str):
        if not name or name == self.label_set or not self.project:
            return
        if not self._confirm_unsaved():
            self.set_combo.blockSignals(True)
            self.set_combo.setCurrentText(self.label_set)
            self.set_combo.blockSignals(False)
            return
        self.label_set = name
        self.refresh_classes()
        self.refresh_list()
        if self.stem:
            self.load_image(self.stem, check_unsaved=False)

    # ================================================================== images
    def _confirm_unsaved(self) -> bool:
        self._flush_paint()
        if not self.dirty or self.stem is None:
            return True
        if self.autosave.isChecked():
            self.save()
            return True
        r = QMessageBox.question(self, "Unsaved changes",
                                 "Save changes to {name} ({set})?".format(name=self.stem, set=self.label_set),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            self.save()
        else:
            self.project.clear_recovery(self.label_set, self.stem)
            self.dirty = False
        return True

    def step_image(self, step: int):
        if not self.project or self.image_list.count() == 0:
            return
        stems = [self.image_list.item(i).data(Qt.UserRole) for i in range(self.image_list.count())]
        i = stems.index(self.stem) + step if self.stem in stems else 0
        if 0 <= i < len(stems):
            self.load_image(stems[i])
        else:
            self.say("No more images in this list.")

    def _reload(self):
        if self.stem:
            if self.project:
                self.project.clear_recovery(self.label_set, self.stem)
            self.dirty = False
            self.load_image(self.stem, check_unsaved=False)

    def _update_search_image(self):
        if self.image_raw is None:
            return
        self.search_image = batch.single_channel(self.image_raw, self.layout_info, self.channel.value())

    @_guard
    def load_image(self, stem: str, check_unsaved: bool = True):
        if check_unsaved and not self._confirm_unsaved():
            return
        path = self.project.image_path(stem)
        if path is None:
            return self.say(f"Image {stem} no longer exists in images/.")
        if any(k[1] == stem for k in self._pending_saves):
            self._wait_saves()                     # never read a label file that is still being written
        try:
            img = io.read_image(path)
        except Exception as e:
            return self.say(f"⚠ Could not open {path.name}: {e}")
        layout = io.guess_layout(img, self.axis_combo.currentData())
        shape = tuple(layout["spatial_shape"])
        previous, same_image = self.stem, stem == self.stem
        camera = (tuple(self.viewer.camera.center), self.viewer.camera.zoom) if same_image else None
        new_shape = shape != self.image_shape

        self._remove_extra_layers()                # overlays of the previous image
        self.problems, self.problem_i = [], -1
        self.problem_label.setText("")
        self.image_raw, self.layout_info = img, layout
        self._show_image(img, layout, path)
        self._update_search_image()
        self.stem, self.image_shape = stem, shape

        labels, classes, source = self.project.load_labels(self.label_set, stem, shape)
        restored = False
        if self.project.has_recovery(self.label_set, stem):
            if self.ask(f"Unsaved work for {path.name} ({self.label_set}) was found – probably from a power cut "
                        f"or crash. Restore it?"):
                labels, classes = self.project.load_recovery(self.label_set, stem, shape)
                source, restored = "recovered backup", True
            else:
                self.project.clear_recovery(self.label_set, stem)
        self.classes = dict(classes)
        self.merge_first = None
        self._show_labels(labels)
        for s in self.project.label_sets:          # other label sets: read-only outlines for context
            if s != self.label_set:
                other, _, _ = self.project.load_labels(s, stem, shape)
                if other.any():
                    self._add_overlay(other, f"{s} (other set)", visible=self.show_other_sets.isChecked(),
                                      below_edit=True)["ani_other_set"] = True
        self._reset_history()
        self.dirty = restored
        self._toggle_prediction(self.show_pred.isChecked())
        self._apply_outline_view(self.outlines_only.isChecked())
        self.set_tool(self.tool if self.tool not in ("New cell", "Outline") else "Inspect")
        self._update_colours()
        if previous and previous != stem:
            self._update_list_item(previous)
        self._update_list_item(stem, select=True)
        if camera:
            self.viewer.camera.center, self.viewer.camera.zoom = camera
        elif new_shape:
            self.viewer.reset_view()
        self._update_title()
        warn = "⚠ " if source.startswith("EMPTY") else ""
        self.say(f"{warn}{path.name} [{self.label_set}]: {len(tools.label_ids(labels))} objects from {source}. "
                 f"Status: {STATUS_TEXT[self.project.status(stem)]}")

    # ---------------------------------------------------------------- viewer layers (reused between images)
    def _show_image(self, img, layout, path):
        """Put the image on screen. Existing image layers are reused when the layout is the same, which
        is much faster than destroying and re-creating them for every image."""
        from napari.layers import Image

        if layout["rgb"]:
            planes, key = [img], ("rgb",)
        elif layout["channel_axis"] is not None:
            planes, key = [img[i] for i in range(img.shape[0])], ("channels", img.shape[0])
        else:
            planes, key = [img], ("grey", img.ndim)
        alive = [lyr for lyr in self._image_layers if lyr in self.viewer.layers]
        if key == self._image_key and len(alive) == len(planes):
            for i, (lyr, plane) in enumerate(zip(alive, planes)):
                lyr.data = plane
                if not layout["rgb"]:
                    lyr.reset_contrast_limits_range()
                    lyr.contrast_limits = io.contrast_limits(plane)
                lyr.name = path.name if len(planes) == 1 else f"{path.stem} ch{i}"
            return
        for lyr in alive:
            self.viewer.layers.remove(lyr)
        cmaps = ["gray", "green", "magenta", "cyan"] if len(planes) > 1 else ["gray"]
        self._image_layers = []
        for i, plane in enumerate(planes):
            if layout["rgb"]:
                lyr = Image(plane, name=path.name, rgb=True)
            else:
                lyr = Image(plane, name=path.name if len(planes) == 1 else f"{path.stem} ch{i}",
                            colormap=cmaps[i % len(cmaps)], blending="additive" if len(planes) > 1 else "translucent",
                            contrast_limits=io.contrast_limits(plane))
            self.viewer.layers.insert(i, lyr)
            self._image_layers.append(lyr)
        self._image_key = key

    def _show_labels(self, labels):
        name = f"{self.label_set} (edit)"
        lyr = self.labels_layer
        if lyr is not None and lyr in self.viewer.layers and lyr.data.ndim == labels.ndim:
            self._busy = True
            try:
                lyr.data = labels
            finally:
                self._busy = False
            lyr.name = name
            lyr.selected_label = 1
            return
        if lyr is not None and lyr in self.viewer.layers:
            self.viewer.layers.remove(lyr)
        lyr = self.viewer.add_labels(labels, name=name, opacity=0.45)
        _slim_history(lyr)
        lyr.mouse_drag_callbacks.append(self._on_click)
        lyr.events.paint.connect(self._on_paint)
        lyr.bind_key("Control-Z", lambda _: self.undo(), overwrite=True)
        lyr.bind_key("Control-Shift-Z", lambda _: self.redo(), overwrite=True)
        self.labels_layer = lyr

    def _add_overlay(self, data, name, visible=True, below_edit=False, contour=1) -> dict:
        """A read-only outline layer (detector output, other label sets). Returns its metadata dict."""
        from napari.layers import Labels

        lyr = Labels(data, name=name, opacity=0.6 if below_edit else 0.8, visible=visible)
        lyr.contour = contour
        lyr.editable = False
        lyr.metadata["ani_overlay"] = True
        if below_edit and self.labels_layer in self.viewer.layers:
            self.viewer.layers.insert(self.viewer.layers.index(self.labels_layer), lyr)
        else:
            self.viewer.layers.append(lyr)
        if self.labels_layer is not None:
            self.viewer.layers.selection.active = self.labels_layer
        return lyr.metadata

    def _remove_extra_layers(self):
        for lyr in [lyr for lyr in self.viewer.layers if lyr.metadata.get("ani_overlay") or lyr is self.problem_layer]:
            self.viewer.layers.remove(lyr)
        self.pred_layer = self.problem_layer = None

    def _toggle_prediction(self, on: bool):
        if not self.project or self.stem is None or self.labels_layer is None:
            return
        if on and self.pred_layer is None:
            try:
                pred, _ = self.project.prediction(self.label_set, self.stem, self.image_shape)
            except Exception as e:
                return self.say("⚠ Detector output unusable: {err}".format(err=e))
            if pred is None:
                return self.say("No detector output for this image.")
            self._add_overlay(pred, "detector output")
            self.pred_layer = self.viewer.layers[-1]
        elif self.pred_layer is not None:
            self.pred_layer.visible = on

    def _toggle_other_sets(self, on: bool):
        for lyr in self.viewer.layers:
            if lyr.metadata.get("ani_other_set"):
                lyr.visible = on

    def _apply_outline_view(self, on: bool):
        if self.labels_layer is not None:
            self.labels_layer.contour = 2 if on else 0
            self.labels_layer.opacity = 0.9 if on else 0.45

    def _update_title(self):
        if self.stem:
            star = "   ● unsaved" if self.dirty else ""
            self.viewer.title = f"Ani – {self.stem} [{self.label_set}]{star}"

    # ================================================================== history (undo / redo)
    def _reset_history(self):
        self._current = (self.labels_layer.data.copy(), dict(self.classes))
        self.undo_stack, self.redo_stack = [], []
        self._paint_pending = False

    def _trim(self):
        while len(self.undo_stack) > 1 and sum(s[0].nbytes for s in self.undo_stack) > UNDO_BUDGET_BYTES:
            self.undo_stack.pop(0)

    def _mark_dirty(self):
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _on_paint(self, *_):
        """Brush, fill and outline edits from napari: grouped into one undo step per stroke."""
        if self._busy:
            return
        self._paint_pending = True
        self._paint_timer.start()
        self._mark_dirty()

    def _flush_paint(self):
        if not self._paint_pending or self.labels_layer is None:
            return
        self._paint_pending = False
        self.undo_stack.append(self._current)
        self.redo_stack.clear()
        self._current = (self.labels_layer.data.copy(), dict(self.classes))
        self._trim()

    def commit(self, new_data: np.ndarray | None = None):
        """Record a change made by an Ani tool. Change self.classes BEFORE calling."""
        self._flush_paint()
        self.undo_stack.append(self._current)
        self.redo_stack.clear()
        if new_data is not None:
            self._set_layer_data(new_data)
            self._current = (new_data.copy(), dict(self.classes))
        else:  # only classes changed: share the (never mutated) private copy
            self._current = (self._current[0], dict(self.classes))
        self._trim()
        self._mark_dirty()
        self._update_colours()

    def _set_layer_data(self, data):
        """Show new label data. Only the changed pixels are sent to the screen when the change is local
        (delete, merge, split, AI click, undo), which keeps big images responsive."""
        layer = self.labels_layer
        old = layer.data
        self._busy = True
        try:
            if (old.shape == data.shape and old.dtype.kind == data.dtype.kind
                    and (data.size == 0 or int(data.max()) <= np.iinfo(old.dtype).max)):
                changed = old != data
                n = int(np.count_nonzero(changed))
                if n == 0:
                    return
                if n < 0.25 * old.size:
                    idx = np.nonzero(changed)
                    layer.data_setitem(idx, data[idx])
                    return
            layer.data = data.copy()
        finally:
            self._busy = False

    def _restore(self, state):
        self._current = state
        self.classes = dict(state[1])
        self._set_layer_data(state[0])
        self._mark_dirty()
        self._update_colours()

    def undo(self):
        if self.labels_layer is None:
            return
        self._flush_paint()
        if not self.undo_stack:
            return self.say("Nothing to undo.")
        self.redo_stack.append(self._current)
        self._restore(self.undo_stack.pop())
        self.say("Undone ({n} more steps available).".format(n=len(self.undo_stack)))

    def redo(self):
        if self.labels_layer is None:
            return
        self._flush_paint()
        if not self.redo_stack:
            return self.say("Nothing to redo.")
        self.undo_stack.append(self._current)
        self._restore(self.redo_stack.pop())
        self.say("Redone.")

    # ================================================================== editing
    @_guard
    def apply(self, fn, *args, msg=""):
        if self.labels_layer is None:
            return
        self._flush_paint()
        self.commit(fn(self.labels_layer.data, *args))
        self.say(msg)

    def set_tool(self, name: str):
        self.tool, self.merge_first = name, None
        for b in self.tool_group.buttons():
            if b.property("tool") == name:
                b.setChecked(True)
        if self.labels_layer is None:
            return
        self.viewer.layers.selection.active = self.labels_layer
        mode = TOOLS[name][0] or "pan_zoom"
        if name in ("New cell", "Outline"):
            self._flush_paint()
            self.labels_layer.selected_label = tools.next_free_label(self.labels_layer.data)
        try:
            self.labels_layer.mode = mode
        except ValueError:  # very old napari without polygon mode
            self.labels_layer.mode = "paint"
        n = self.labels_layer.selected_label
        hints = {"New cell": "Painting NEW object #{n}. Press Shift-N again before the next one.".format(n=n),
                 "Outline": ("Click around the missed object, double-click to close (object #{n}). "
                               "Shift-O again for the next one.").format(n=n),
                 "AI click": "Click inside a missed object. Detector: {name}".format(name=self.backend().name),
                 "Merge": "Click the object to KEEP, then the one to merge into it.",
                 "Split": "Click an object to split it into {n} parts.".format(n=self.split_parts.value()),
                 "Delete": "Click false detections to delete them.",
                 "Set class": "Click objects to mark them '{cls}'.".format(cls=self.class_combo.currentText())}
        self.say(hints.get(name, "Tool: {name}".format(name=name)))

    def _value_at(self, layer, event):
        try:
            v = layer.get_value(event.position, view_direction=event.view_direction,
                                dims_displayed=event.dims_displayed, world=True)
        except TypeError:
            v = layer.get_value(event.position, world=True)
        if isinstance(v, tuple):  # multiscale returns (level, value)
            v = v[-1]
        return int(v) if v is not None else 0

    def _on_click(self, layer, event):
        if event.type != "mouse_press" or event.button != 1 or self.tool not in CLICK_TOOLS:
            return
        if self.tool == "AI click":
            coords = np.round(np.asarray(layer.world_to_data(event.position))).astype(int)
            self.ai_click(tuple(coords[-layer.data.ndim:]))
            return
        self.click_cell(self._value_at(layer, event))

    @_guard
    def click_cell(self, lab: int):
        """Run the current click tool on object `lab` (separate from mouse handling so it is testable)."""
        if self.labels_layer is None:
            return
        if lab == 0:
            if self.tool != "Inspect":
                self.say("That is background – click inside an object.")
            return
        self._flush_paint()
        data = self.labels_layer.data
        default = self.project.classes(self.label_set)[0]
        if self.tool == "Inspect":
            info = tools.region_info(data, lab)
            self.say("Object {n}: {area} px, class '{cls}'".format(n=lab, area=info["area"],
                                                                        cls=self.classes.get(lab, default)))
        elif self.tool == "Delete":
            self.classes.pop(lab, None)
            self.commit(tools.delete_label(data, lab))
            self.say("Deleted object {n}.".format(n=lab))
        elif self.tool == "Split":
            old_max = int(data.max())
            new_data = tools.split_label(data, lab, self.split_parts.value())
            new = range(old_max + 1, int(new_data.max()) + 1)
            if not new:
                return self.say("Object {n} is too small to split.".format(n=lab))
            for n in new:
                if lab in self.classes:
                    self.classes[n] = self.classes[lab]
            self.commit(new_data)
            self.say("Split object {n} into {k} parts.".format(n=lab, k=len(new) + 1))
        elif self.tool == "Merge":
            if self.merge_first is None:
                self.merge_first = lab
                self.say("Object {n} selected. Now click the object to merge into it.".format(n=lab))
            elif lab != self.merge_first:
                keep, self.merge_first = self.merge_first, None
                self.classes.pop(lab, None)
                self.commit(tools.merge_labels(data, keep, lab))
                self.say("Merged object {a} into {b}.".format(a=lab, b=keep))
        elif self.tool == "Set class":
            cls = self.class_combo.currentText()
            if cls == default:
                self.classes.pop(lab, None)
            else:
                self.classes[lab] = cls
            self.commit()
            self.say("Object {n} → {cls}".format(n=lab, cls=cls))

    def _renumber(self):
        if self.labels_layer is None:
            return
        self._flush_paint()
        data, self.classes = tools.relabel_sequential(self.labels_layer.data, self.classes)
        self.commit(data)
        self.say("Renumbered: {n} objects.".format(n=len(tools.label_ids(data))))

    def _clear_all(self):
        if self.labels_layer is None:
            return
        if self.ask("Remove every object of this label set on this image? (Undo can bring them back.)"):
            self.classes = {}
            self.apply(np.zeros_like, msg="Cleared. Use Undo to bring them back.")

    def _update_colours(self):
        layer = self.labels_layer
        if layer is None or not self.project:
            return
        try:
            from napari.utils.colormaps import DirectLabelColormap, label_colormap
        except ImportError:
            return
        if not self.colour_by_class.isChecked():
            if isinstance(layer.colormap, DirectLabelColormap):
                layer.colormap = label_colormap(49)
            return
        names = self.project.classes(self.label_set)
        cls_col = {c: CLASS_COLOURS[i % len(CLASS_COLOURS)] for i, c in enumerate(names)}
        colours = {None: cls_col[names[0]], 0: "transparent"}
        for lab, c in self.classes.items():
            colours[int(lab)] = cls_col.get(c, cls_col[names[0]])
        layer.colormap = DirectLabelColormap(color_dict=colours)

    # ================================================================== AI: this image
    def _ai_params(self) -> dict:
        from .ai.base import polarity_from_labels

        p = dict(self.project.meta.get("classic_params") or {}) if self.project else {}
        if self.diameter.value():
            p["diameter"] = float(self.diameter.value())
        choice = self.polarity.currentData()
        if choice != "auto":
            p["polarity"] = choice
        elif p.get("polarity", "auto") == "auto" and self.labels_layer is not None:
            learned = polarity_from_labels(self.search_image, self.labels_layer.data)
            if learned:
                p["polarity"] = learned
        return p

    def _token(self):
        return (self.stem, self.label_set)

    @_guard
    def auto_label(self, mode: str):
        if self.labels_layer is None or self.search_image is None:
            return
        b = self.backend()
        ok, msg = b.available()
        if not ok or not b.can_auto:
            return self.say(f"✗ {b.name}: " + (msg if not ok else "cannot label whole images"))
        token, plane, params = self._token(), self.search_image, self._ai_params()
        self.say("{name} is labelling this image… (CPU: may take a minute)".format(name=b.name))

        def done(mask):
            if token != self._token() or self.labels_layer is None:
                return self.say("Result ignored: you moved to another image.")
            mask = np.asarray(mask)
            if mask.shape != self.labels_layer.data.shape:
                return self.say("⚠ Detector returned a mask of the wrong size.")
            self._flush_paint()
            if mode == "replace":
                self.classes = {}
                new = mask
            else:
                new = tools.add_new_objects(self.labels_layer.data, mask)
            added = len(tools.label_ids(new)) - (0 if mode == "replace" else len(tools.label_ids(self._current[0])))
            self.commit(io.to_label_dtype(new))
            self.say("{name}: {n} objects added. Check them with 'Find problems'.".format(name=b.name, n=added))

        self.start_job("local", b.auto_segment, plane, params, on_done=done)

    @_guard
    def ai_click(self, point: tuple):
        if self.labels_layer is None or self.search_image is None:
            return
        data = self.labels_layer.data
        if any(c < 0 or c >= n for c, n in zip(point, data.shape)):
            return
        b = self.backend()
        fallback = ""
        if not (b.can_click and b.available()[0]):
            b = self.backends[0]           # Classic always works
            fallback = f" ({b.name})"
        plane, lab = self.search_image, data
        z = None
        if data.ndim == 3:                  # z-stack: work on the slice that was clicked
            z = point[0]
            plane, lab, point = plane[z], data[z], point[1:]
        token, params = self._token(), self._ai_params()
        lab_copy = lab.copy()
        self.say("AI is outlining the object…")

        def done(res):
            mask, box = res if res else (None, None)
            if token != self._token() or self.labels_layer is None:
                return
            if mask is None:
                return self.say("Nothing found here. Try the Outline tool (Shift-O).")
            self._flush_paint()
            new = self.labels_layer.data.copy()
            target = new[z] if z is not None else new
            region = target[box]
            new_id = tools.next_free_label(new)
            region[mask & (region == 0)] = new_id
            self.commit(new)
            self.say("Added object {n} ({area} px).".format(n=new_id, area=int(mask.sum())) + fallback)

        self.start_job("local", b.segment_at, plane, point, lab_copy, params, on_done=done)

    # ================================================================== AI: all images
    @_guard
    def run_batch(self):
        if not self.project:
            return
        self._save_and_wait()
        stems = [p.stem for p in self.project.images()]
        if not stems:
            return self.say("No images in this project yet.")
        if self.use_command.isChecked():
            command = self.command_edit.text().strip()
            if "{input}" not in command or "{output}" not in command:
                return self.say("The command must contain {input} and {output}.")
            self._save_command()
            kwargs = {"command": command}
            name = "your program"
        else:
            idx = self.backend_combo.currentIndex()
            b = ai.backends_for(self.project)[idx]   # own instance: never shared with the UI thread
            ok, msg = b.available()
            if not ok or not b.can_auto:
                return self.say(f"✗ {b.name}: " + (msg if not ok else "cannot label whole images"))
            kwargs = {"backend": b, "params": self._ai_params(), "channel": self.channel.value()}
            name = b.name
        self.batch_progress.setRange(0, len(stems))
        self.batch_progress.setValue(0)
        failures = []

        def progress(item):
            i, n, stem, msg = item
            self.batch_progress.setValue(i + 1)
            self.batch_log.setText(f"{i + 1}/{n} {stem}: {msg}")
            if msg.startswith("FAILED"):
                failures.append(f"{stem}: {msg}")

        def done(_):
            self.refresh_list()
            summary = "Detector finished." + (" " + "{n} image(s) failed.".format(n=len(failures))
                                                  if failures else "")
            self.batch_log.setText(summary + ("\n" + "\n".join(failures[:5]) if failures else ""))
            self.say(summary)
            if self.stem and not self.dirty and not self.project.label_path(self.label_set, self.stem).exists():
                self.load_image(self.stem, check_unsaved=False)   # show the new detector output

        self.say("Running {name} on {n} images in the background. You can keep annotating."
                 .format(name=name, n=len(stems)))
        self.start_job("batch", batch.run_detector, self.project, self.label_set, stems,
                       overwrite=self.overwrite.isChecked(), on_yield=progress, on_done=done, **kwargs)

    @_guard
    def rank_all(self):
        if not self.project:
            return
        self._save_and_wait()
        stems = [p.stem for p in self.project.images()]
        self.batch_progress.setRange(0, max(1, len(stems)))

        def progress(item):
            i, n, stem, msg = item
            self.batch_progress.setValue(i + 1)
            self.batch_log.setText(f"{i + 1}/{n} {stem}: {msg}")

        def done(_):
            self.sort_combo.setCurrentIndex(self.sort_combo.findData("worst"))
            self.refresh_list()
            self.batch_log.setText("Images ranked: the most suspicious are at the top of the list.")

        self.start_job("batch", batch.rank_images, self.project, self.label_set, stems,
                       channel=self.channel.value(), on_yield=progress, on_done=done)

    # ================================================================== Train tab
    def _refresh_eval_table(self):
        rows = (self.project.meta.get("evaluations") or []) if self.project else []
        rows = rows[-20:][::-1]
        self.eval_table.setRowCount(len(rows))
        for r, e in enumerate(rows):
            vals = [e["when"], f"{e['model']} [{e['label_set']}]", e["images"], e["precision"], e["recall"],
                    e["f1"], e["mean_iou"]]
            for c, v in enumerate(vals):
                self.eval_table.setItem(r, c, QTableWidgetItem(f"{v:.3f}" if isinstance(v, float) else str(v)))
        self.eval_table.resizeColumnsToContents()

    @_guard
    def evaluate(self):
        if not self.project:
            return
        from . import train

        self._save_and_wait()
        backend = None
        if self.eval_source.currentData() == "backend":
            idx = self.backend_combo.currentIndex()
            backend = ai.backends_for(self.project)[idx]
            ok, msg = backend.available()
            if not ok or not backend.can_auto:
                return self.say(f"✗ {backend.name}: " + (msg if not ok else "cannot label whole images"))

        def done(res):
            o = res["overall"]
            self._refresh_eval_table()
            self.train_log.setText("{model}: F1 {f1:.3f} (precision {p:.3f}, recall {r:.3f}) on {n} images."
                                   .format(model=o["model"], f1=o["f1"], p=o["precision"], r=o["recall"],
                                           n=o["images"]))
            self.say(self.train_log.text())

        self.train_log.setText("Evaluating…")
        self.start_job("train", train.evaluate, self.project, self.label_set, backend, self._ai_params(),
                       channel=self.channel.value(), on_done=done)

    @_guard
    def tune(self):
        if not self.project:
            return
        from . import train

        self._save_and_wait()
        self._tune_result = None
        self.train_progress.setRange(0, self.tune_trials.value())

        def progress(item):
            if item[0] == "progress":
                _, i, n, best = item
                self.train_progress.setValue(i)
                self.train_log.setText("Trying settings {i}/{n} – best F1 so far {f1:.3f}".format(
                    i=i, n=n, f1=best))
            else:
                self._tune_result = item

        def done(_):
            if not self._tune_result:
                return
            _, best, f1, old = self._tune_result
            self.refresh_backends()
            if f1 > old:
                msg = "Classic improved: F1 {old:.3f} → {new:.3f}. New settings saved.".format(old=old, new=f1)
            else:
                msg = "Current settings are already the best found (F1 {f1:.3f}).".format(f1=old)
            self.train_log.setText(msg)
            self.say(msg)

        self.start_job("train", train.tune_classic, self.project, self.label_set, self.tune_trials.value(),
                       channel=self.channel.value(), on_yield=progress, on_done=done)

    @_guard
    def finetune(self):
        if not self.project:
            return
        from . import train

        self._save_and_wait()
        self.train_progress.setRange(0, 0)   # busy indicator

        def done(path):
            self.refresh_backends()
            msg = "New Cellpose model saved: {name}. Choose it in the AI tab and press Evaluate.".format(
                name=Path(path).name)
            self.train_log.setText(msg)
            self.say(msg)

        self.train_log.setText("Training Cellpose on CPU – this can take a while. You can keep annotating.")
        self.start_job("train", train.finetune_cellpose, self.project, self.label_set, self.ft_base.currentText(),
                       self.ft_epochs.value(), channel=self.channel.value(), on_done=done)

    # ================================================================== find problems
    def _typical_size(self) -> float:
        areas = np.bincount(self._current[0].ravel())[1:]
        areas = areas[areas > 0]
        return math.sqrt(float(np.median(areas))) if len(areas) else 15.0

    @_guard
    def find_problems(self):
        """Look for likely mistakes in a background thread, so annotation is never blocked."""
        if self.labels_layer is None:
            return
        self._flush_paint()
        data = self._current[0]                       # private copy: safe to read from another thread
        image = self.search_image if self.search_missed.isChecked() else None
        token = self._token()
        self.say("Looking for problems…")

        def work():
            found = tools.find_suspects(data)
            if image is not None and image.shape == data.shape:
                found += tools.find_missed(image, data)
            return sorted(found, key=lambda d: -d["score"])

        def done(found):
            if token != self._token() or self.labels_layer is None:
                return
            self._show_problems(found)

        self.start_job("local", work, on_done=done)

    def _show_problems(self, found):
        self.problems, self.problem_i = found, -1
        if self.problem_layer is not None and self.problem_layer in self.viewer.layers:
            self.viewer.layers.remove(self.problem_layer)
        self.problem_layer = None
        if not self.problems:
            self.problem_label.setText("No problems found 👍")
            return self.say("No suspicious objects found.")
        typical = self._typical_size()
        pts = np.array([p["centroid"] for p in self.problems])
        sizes = [max(12.0, 2.2 * math.sqrt(p.get("area", typical ** 2))) for p in self.problems]
        self.problem_layer = self.viewer.add_points(pts, name="problems", size=sizes, symbol="ring",
                                                    face_color="transparent")
        self._colour_problems()
        self.viewer.layers.selection.active = self.labels_layer
        n_missed = sum(p["label"] == 0 for p in self.problems)
        self.say(f"Found {len(self.problems)} possible problems ({n_missed} possibly missed). Use ▶ to visit them.")
        self.goto_problem(1)

    def _colour_problems(self):
        if self.problem_layer is None:
            return
        cols = np.array([[1, 0.85, 0.2, 1]] * len(self.problems))
        if 0 <= self.problem_i < len(self.problems):
            cols[self.problem_i] = [1, 0.2, 0.9, 1]
        attr = "border_color" if hasattr(self.problem_layer, "border_color") else "edge_color"
        setattr(self.problem_layer, attr, cols)

    def goto_problem(self, step: int):
        if not self.problems:
            return self.say("Press 'Find problems' first.")
        self.problem_i = (self.problem_i + step) % len(self.problems)
        p = self.problems[self.problem_i]
        c = list(p["centroid"])
        if len(c) == 3:
            self.viewer.dims.set_point(0, c[0])
            c = c[1:]
        self.viewer.camera.center = tuple(c)
        canvas = min(getattr(self.viewer, "_canvas_size", (800, 800)))
        self.viewer.camera.zoom = max(0.5, canvas / max(10 * self._typical_size(), 80))
        self._colour_problems()
        what = "Object {n}".format(n=p["label"]) if p["label"] else "Unlabelled area"
        hint = "Use AI click, Outline or New cell to add it, or skip." if not p["label"] else \
            "Fix with Delete / Split / Merge / Paint, or skip if it is fine."
        self.problem_label.setText(f"<b>{self.problem_i + 1}/{len(self.problems)}</b> – {what}: "
                                   f"{p['reason']}<br><span style='color:#999'>{hint}</span>")
        if p["label"]:
            self.labels_layer.selected_label = p["label"]

    # ================================================================== save / backup / export
    @_guard
    def save(self):
        """Save the current image. The disk write happens in the background so annotation never waits."""
        if self.labels_layer is None or self.stem is None:
            return
        self._flush_paint()
        data, classes = self._current               # private copy, never changed afterwards
        project, label_set, stem, shape = self.project, self.label_set, self.stem, self.image_shape
        self.dirty = False
        self._update_title()

        def work():
            try:
                return project.save_labels(label_set, stem, data, classes, shape)
            except Exception:
                project.write_recovery(label_set, stem, data, classes)   # keep the work safe
                raise

        def done(stats):
            self._update_list_item(stem)
            self.say(f"Saved {stem} [{label_set}]: {stats['cells']} objects | vs detector: +{stats['added']} "
                     f"added, −{stats['removed']} removed, {stats['modified']} reshaped")

        self._submit_save((label_set, stem), work, done)

    def _submit_save(self, key, work, done=None):
        if self.sync_jobs:                           # tests: run immediately
            res = work()
            if done:
                done(res)
            return
        self._pending_saves.add(key)
        fut = self._saver.submit(work)
        fut.add_done_callback(lambda f: self._bridge.done.emit((key, f, done)))

    def _save_finished(self, payload):
        key, fut, done = payload
        self._pending_saves.discard(key)
        try:
            res = fut.result()
        except Exception as e:
            self.say(f"⚠ Saving {key[1]} FAILED: {e}. Your work was kept as a backup and will be offered "
                     f"when you open the image again.")
            return
        if done:
            self._safe(done)(res)

    def _save_and_wait(self):
        if self.dirty:
            self.save()
        self._wait_saves()

    def _wait_saves(self):
        """Block until every queued save is on disk (before reading label files for export, training…)."""
        self._saver.submit(lambda: None).result()
        QApplication.processEvents()

    def backup_now(self):
        """Called every minute: write unsaved work to .recovery/ (survives power cuts) – in the background."""
        if not (self.dirty and self.project and self.stem and self.labels_layer is not None):
            return
        self._flush_paint()
        project, label_set, stem = self.project, self.label_set, self.stem
        data, classes = self._current
        self._submit_save(("backup", stem), lambda: project.write_recovery(label_set, stem, data, classes))

    def _on_quit(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._backup_timer.stop()
        for w in list(self.jobs.values()):
            if hasattr(w, "quit"):
                w.quit()
        from qtpy.QtCore import QThreadPool
        QThreadPool.globalInstance().waitForDone(15000)   # let running AI / problem jobs finish cleanly
        try:
            if self.dirty and self.project and self.stem and self._current is not None:
                if self.autosave.isChecked():
                    self.project.save_labels(self.label_set, self.stem, *self._current, self.image_shape)
                else:
                    self.project.write_recovery(self.label_set, self.stem, *self._current)
        except Exception:
            pass
        self._saver.shutdown(wait=True)              # finish queued saves before the app exits

    def _set_status(self, status):
        if self.stem is None:
            return
        self.project.set_status(self.stem, status)
        self._update_list_item(self.stem)
        self.say(f"{self.stem} marked {STATUS_TEXT[status]}.")

    def save_done_next(self):
        if self.stem is None:
            return
        self.save()
        self._set_status("done")
        todo = [p.stem for p in self.project.images() if self.project.status(p.stem) in ("todo", "in progress")]
        if todo:
            self.load_image(todo[0], check_unsaved=False)
        else:
            self.say("🎉 All images are done! Export your training dataset in the Train tab.")

    @_guard
    def _export(self):
        if not self.project:
            return
        self._save_and_wait()
        from .export import export_dataset
        from .train import write_notebooks

        def progress(i, n, stem):
            self.say("Exporting {i}/{n}: {name}".format(i=i, n=n, name=stem))
            QApplication.processEvents()

        try:
            out = export_dataset(self.project, formats=[k for k, cb in self.fmt.items() if cb.isChecked()],
                                 only_done=self.only_done.isChecked(), val_fraction=self.val_pct.value() / 100,
                                 convert_8bit=self.to_png.isChecked(), progress=progress)
        except ValueError as e:
            return self.say(str(e))
        if self.with_notebooks.isChecked():
            write_notebooks(out, self.label_set)
        self.say("✔ Exported to {path}".format(path=out))
