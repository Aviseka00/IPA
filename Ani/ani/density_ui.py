"""The densitogram window: profile, notch baseline, peak table, and the 3D switch."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .density import DensityResult, analyze_lane, write_density_csv

_PEAK_COLORS = ("#ff5c5c", "#e86bff", "#ffb020", "#3ddc97", "#7aa2ff")


class _ProfilePlot(QWidget):
    """The density trace, with each peak filled down to the notch line."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self._result: DensityResult | None = None

    def set_result(self, result: DensityResult | None):
        self._result = result
        self.update()

    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171a21"))
        result = self._result
        if result is None or result.profile.size < 2:
            painter.setPen(QColor("#a7b0be"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "Mark an area, then click Generate")
            painter.end()
            return
        profile = result.profile
        left, top, right, bottom = 44, 12, self.width() - 12, self.height() - 26
        if right - left < 8 or bottom - top < 8:
            painter.end()
            return
        ymax = max(float(profile.max()) * 1.08, 1e-6)
        span_x = max(profile.size - 1, 1)

        def x_of(sample: float) -> float:
            return left + (sample / span_x) * (right - left)

        def y_of(value: float) -> float:
            return bottom - (float(value) / ymax) * (bottom - top)

        painter.setPen(QPen(QColor("#3c4454"), 1))
        painter.drawLine(int(left), int(bottom), int(right), int(bottom))
        painter.drawLine(int(left), int(top), int(left), int(bottom))
        for peak in result.peaks:
            color = QColor(_PEAK_COLORS[(peak.number - 1) % len(_PEAK_COLORS)])
            color.setAlpha(150)
            polygon = QPolygonF()
            for sample in range(peak.left, peak.right + 1):
                polygon.append(self._point(x_of(sample), y_of(profile[sample])))
            for sample in range(peak.right, peak.left - 1, -1):
                base = result.baseline[sample]
                polygon.append(self._point(x_of(sample), y_of(0.0 if np_isnan(base) else base)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(polygon)

        notch = QPen(QColor("#39e0ff"), 1.4)
        notch.setStyle(Qt.DashLine)
        painter.setPen(notch)
        painter.setBrush(Qt.NoBrush)
        for peak in result.peaks:
            painter.drawLine(int(x_of(peak.left)), int(y_of(profile[peak.left])),
                             int(x_of(peak.right)), int(y_of(profile[peak.right])))

        painter.setPen(QPen(QColor("#f4f6f8"), 1.6))
        for sample in range(profile.size - 1):
            painter.drawLine(int(x_of(sample)), int(y_of(profile[sample])),
                             int(x_of(sample + 1)), int(y_of(profile[sample + 1])))

        painter.setFont(QFont("Segoe UI", 8))
        for peak in result.peaks:
            painter.setPen(QColor(_PEAK_COLORS[(peak.number - 1) % len(_PEAK_COLORS)]))
            painter.drawText(int(x_of(peak.sample)) - 6, int(y_of(profile[peak.sample])) - 6, str(peak.number))

        painter.setPen(QColor("#a7b0be"))
        painter.drawText(4, int(top) + 10, "Density")
        painter.drawText(int(left), self.height() - 6, "Position along the lane")
        painter.end()

    @staticmethod
    def _point(x: float, y: float) -> QPointF:
        return QPointF(x, y)


def np_isnan(value) -> bool:
    return value != value


class DensityDialog(QDialog):
    """Peak intensity, area and spread for the lane the user dragged."""

    def __init__(self, annotator):
        super().__init__(annotator)
        self.annotator = annotator
        self.setWindowTitle("Densitogram")
        self.setMinimumWidth(520)
        self._result: DensityResult | None = None
        self._results: list[DensityResult] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        title = QLabel("Densitogram")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QLabel(
            "Each outlined spot is one area. ROI numbers match the 3D peaks. "
            "Reanalyze updates the graph without redrawing. Lock notch keeps the baseline fixed."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        root.addWidget(hint)

        self.direction = QComboBox()
        self.direction.addItem("Auto (row or lane)", "auto")
        self.direction.addItem("Down the lane", "vertical")
        self.direction.addItem("Across the lane", "horizontal")
        self.bands = QComboBox()
        self.bands.addItem("Dark bands", True)
        self.bands.addItem("Bright bands", False)
        menus = QHBoxLayout()
        menus.setSpacing(6)
        menus.addWidget(self.direction, 1)
        menus.addWidget(self.bands, 1)
        root.addLayout(menus)
        self.direction.currentIndexChanged.connect(lambda *_: self.refresh())
        self.bands.currentIndexChanged.connect(lambda *_: self.refresh())

        self.notch_label = QLabel("Notch depth 8%")
        self.notch_label.setObjectName("hint")
        self.notch_slider = QSlider(Qt.Horizontal)
        self.notch_slider.setRange(2, 40)
        self.notch_slider.setValue(5)
        self.notch_slider.setToolTip("Higher keeps only deeper troughs, so shallow valleys do not split a peak")
        self.lock_notch = QCheckBox("Lock notch")
        self.lock_notch.setToolTip("Keep this notch depth when you reanalyze")
        self.lock_notch.toggled.connect(self._lock_changed)
        notch = QHBoxLayout()
        notch.setSpacing(8)
        self.notch_label.setFixedWidth(118)
        notch.addWidget(self.notch_label)
        notch.addWidget(self.notch_slider, 1)
        notch.addWidget(self.lock_notch)
        root.addLayout(notch)
        self.notch_slider.valueChanged.connect(self._notch_moved)

        self.roi_choice = QComboBox()
        self.roi_choice.setToolTip("Which marked area the graph shows. Same ROI number as in 3D and the CSV.")
        self.roi_choice.currentIndexChanged.connect(lambda *_: self._show_selected())
        root.addWidget(self.roi_choice)

        self.plot = _ProfilePlot()
        root.addWidget(self.plot)
        self.summary = QLabel("Mark an area, then click Generate.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("summary")
        root.addWidget(self.summary)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ROI", "Peak", "Intensity", "Height", "Area", "Spread"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMaximumHeight(168)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setToolTip(
            "Intensity is Fiji-style optical density at the band core. "
            "Height is rise above the notch. Area is integrated density between notches."
        )
        root.addWidget(self.table)

        turn = QPushButton("3D view")
        turn.setObjectName("primary")
        turn.setToolTip("Turn the outlined ROIs. Peak numbers match this table.")
        turn.clicked.connect(lambda: self.annotator.show_density_3d())
        flat = QPushButton("Normal")
        flat.setToolTip("Put the picture back in its usual flat position, facing you")
        flat.clicked.connect(lambda: self.annotator.show_density_2d())
        reanalyze = QPushButton("Reanalyze")
        reanalyze.setToolTip("Rebuild the graph and table from the current outlines, without redrawing")
        reanalyze.clicked.connect(self.reanalyze)
        export = QPushButton("Export CSV")
        export.setToolTip("Save the peak table for every outlined area")
        export.clicked.connect(self.export_csv)
        profile = QPushButton("Export profile")
        profile.setToolTip("Save a PNG of the densitogram graph")
        profile.clicked.connect(self.export_profile)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        row = QHBoxLayout()
        row.setSpacing(6)
        for button in (turn, flat, reanalyze):
            row.addWidget(button, 1)
        root.addLayout(row)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        for button in (export, profile, close):
            row2.addWidget(button, 1)
        root.addLayout(row2)
        self.setStyleSheet("""
            QDialog { background: #171a21; color: #f4f6f8; font-family: "Segoe UI", sans-serif; font-size: 13px; }
            QLabel { color: #f4f6f8; background: transparent; }
            QLabel#title { font-size: 18px; font-weight: 600; }
            QLabel#hint { color: #a7b0be; font-size: 12px; }
            QLabel#summary { font-size: 13px; font-weight: 600; }
            QPushButton {
                background: #232833; color: #f4f6f8; border: 1px solid #3c4454;
                border-radius: 7px; padding: 4px 8px; min-height: 28px; font-size: 13px;
            }
            QPushButton:hover { border-color: #6f8cff; }
            QPushButton#primary { background: #6f8cff; color: #12141a; border: none; font-weight: 600; }
            QComboBox, QTableWidget, QCheckBox {
                background: #232833; color: #f4f6f8; border: 1px solid #3c4454;
                border-radius: 7px; padding: 4px 8px; font-size: 13px;
            }
            QCheckBox { border: none; background: transparent; }
            QHeaderView::section { background: #232833; color: #a7b0be; border: none; padding: 4px; font-size: 12px; }
            QTableWidget { padding: 0; }
            QSlider::groove:horizontal { height: 4px; background: #3c4454; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #6f8cff; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }
        """)

    def _lock_changed(self, locked: bool):
        self.notch_slider.setEnabled(not locked)
        if locked:
            self.summary.setText(f"Notch locked at {int(self.notch_slider.value())}%. Reanalyze keeps this baseline.")

    def _notch_moved(self, value: int):
        self.notch_label.setText(f"Notch depth {int(value)}%")
        if not self.lock_notch.isChecked():
            self.refresh()

    def reanalyze(self):
        """Rebuild from current outlines without asking the user to redraw."""
        self.annotator.flush()
        self.annotator._rois = self.annotator._rois_from_outlines()
        self.refresh()
        if self._results:
            self.summary.setText(
                f"Reanalyzed {len(self._results)} ROI(s) without redrawing. "
                "ROI numbers match the 3D peaks and the CSV."
            )
            self.annotator._set_status(f"Densitogram ready for {len(self._results)} outlined areas.")
        else:
            self.summary.setText("No outlines to reanalyze. Draw spots, then click Generate.")

    def refresh(self):
        image = self.annotator._display_image
        rois = list(self.annotator._rois)
        if image is None or not rois:
            self._result = None
            self._results = []
            self.plot.set_result(None)
            self.roi_choice.blockSignals(True)
            self.roi_choice.clear()
            self.roi_choice.blockSignals(False)
            self.summary.setText("Draw the spots first, then click Generate.")
            self.table.setRowCount(0)
            return
        chosen = self.roi_choice.currentIndex()
        self._results = [
            analyze_lane(
                image,
                roi=item["roi"],
                along=str(self.direction.currentData()),
                dark_bands=bool(self.bands.currentData()),
                notch_percent=float(self.notch_slider.value()),
                mask=self.annotator._roi_mask(item),
            )
            for item in rois
        ]
        if self._results:
            sample = self._results[0]
            self.annotator._density_along = sample.along
            self.annotator._density_dark = sample.dark_bands
            self.annotator._density_notch = sample.notch_percent
        self.roi_choice.blockSignals(True)
        self.roi_choice.clear()
        for number, item in enumerate(rois, start=1):
            kind = item.get("name") or ("spot" if item["kind"] == "cell" else "area")
            self.roi_choice.addItem(f"ROI {number} · {kind}")
        if self.roi_choice.count():
            self.roi_choice.setCurrentIndex(chosen if 0 <= chosen < self.roi_choice.count() else self.roi_choice.count() - 1)
        self.roi_choice.blockSignals(False)
        self._show_selected()
        self._fill_table()
        try:
            self.annotator._show_roi_boxes()
        except Exception:
            pass

    def _show_selected(self):
        index = self.roi_choice.currentIndex()
        if index < 0 or index >= len(self._results):
            self._result = None
            self.plot.set_result(None)
            return
        self._result = self._results[index]
        self.plot.set_result(self._result)
        rois = self.annotator._rois
        if 0 <= index < len(rois):
            item = rois[index]
            self.annotator._lane_roi = item["roi"]
            self.annotator._density_mask = item["mask"]
            self.annotator._density_kind = item["kind"]
        peaks = self._result.peaks
        if not peaks:
            self.summary.setText(f"ROI {index + 1} has no peaks. Lower the notch depth if a shallow trough should count.")
            return
        tallest = max(peaks, key=lambda peak: peak.intensity)
        self.summary.setText(
            f"ROI {index + 1}: {len(peaks)} peaks. "
            f"Tallest is peak {tallest.number}, height {tallest.intensity:.0f}, "
            f"area {tallest.area:.0f}, intensity {tallest.level:.0f}."
        )

    def _fill_table(self):
        rows = [(number, peak) for number, result in enumerate(self._results, start=1) for peak in result.peaks]
        self.table.setRowCount(len(rows))
        for row, (number, peak) in enumerate(rows):
            values = [
                str(number),
                str(peak.number),
                f"{peak.level:.1f}",
                f"{peak.intensity:.1f}",
                f"{peak.area:.0f}",
                f"{peak.spread:.0f} px",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        if not rows and self._results:
            self.summary.setText("No peaks in the marked areas. Lower the notch depth if a shallow trough should count.")

    def export_csv(self) -> bool:
        if not self._results:
            self.summary.setText("Generate the densitogram first, then export.")
            return False
        start = ""
        if self.annotator.image_path is not None:
            start = str(Path(self.annotator.image_path).with_name(f"{self.annotator.image_path.stem}_density.csv"))
        path, _ = QFileDialog.getSaveFileName(self, "Export densitogram", start, "CSV (*.csv)")
        if not path:
            return False
        names = []
        for item in self.annotator._rois:
            names.append(item.get("name") or ("spot" if item.get("kind") == "cell" else "area"))
        written = write_density_csv(path, self._results, names)
        self.summary.setText(f"Saved peak table to {written.name}.")
        return True

    def export_profile(self) -> bool:
        if self._result is None:
            self.summary.setText("Generate the densitogram first, then export the profile.")
            return False
        start = ""
        if self.annotator.image_path is not None:
            start = str(Path(self.annotator.image_path).with_name(f"{self.annotator.image_path.stem}_profile.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Export densitogram profile", start, "PNG (*.png)")
        if not path:
            return False
        image = self.plot.grab()
        if not image.save(path, "PNG"):
            self.summary.setText("Could not save the profile image.")
            return False
        self.summary.setText(f"Saved profile graph to {Path(path).name}.")
        return True

    def current_results(self) -> list[DensityResult]:
        return list(self._results)
