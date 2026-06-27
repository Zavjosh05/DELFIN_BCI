"""Visor multicanal de señales EEG basado en pyqtgraph.

Muestra los canales apilados con desplazamiento vertical, permite alternar entre
señal cruda y procesada, navegar en el tiempo y seleccionar una región para
crear un segmento etiquetado (agrupar/aislar señales para el dataset).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

pg.setConfigOptions(antialias=False, background="#101317", foreground="#c8d0d8")

_CURVE_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8", "#4DD0E1",
    "#AED581", "#FFD54F", "#F06292", "#9575CD", "#4DB6AC", "#DCE775",
    "#FF8A65", "#90A4AE",
]


class SignalView(QWidget):
    segment_requested = pyqtSignal(int, int)  # (start_sample, stop_sample)
    mode_changed = pyqtSignal()               # cambió Cruda/Procesada

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None       # (n_canales, n_muestras)
        self._fs: float = 128.0
        self._channel_names: list[str] = []
        self._spacing: float = 0.0

        self._build_ui()

    # --- Construcción de la interfaz --------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Procesada", "Cruda"])
        self.mode_box.setToolTip("Mostrar la señal con o sin el pipeline aplicado")
        controls.addWidget(QLabel("Vista:"))
        controls.addWidget(self.mode_box)

        self.gain_box = QComboBox()
        self.gain_box.addItems(["x0.25", "x0.5", "x1", "x2", "x4", "x8"])
        self.gain_box.setCurrentText("x1")
        self.gain_box.currentTextChanged.connect(self._redraw)
        controls.addWidget(QLabel("Ganancia:"))
        controls.addWidget(self.gain_box)

        self.norm_chk = QCheckBox("Normalizar vista")
        self.norm_chk.setChecked(True)
        self.norm_chk.stateChanged.connect(self._redraw)
        controls.addWidget(self.norm_chk)

        controls.addStretch(1)
        self.sel_label = QLabel("Selección: —")
        controls.addWidget(self.sel_label)
        self.add_seg_btn = QPushButton("Crear segmento de la selección")
        self.add_seg_btn.clicked.connect(self._emit_segment)
        self.add_seg_btn.setEnabled(False)
        controls.addWidget(self.add_seg_btn)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        layout.addWidget(self.plot, 1)

        # Región de selección de tiempo.
        self.region = pg.LinearRegionItem(brush=(80, 160, 255, 40))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.hide()

        self.mode_box.currentTextChanged.connect(lambda *_: self.mode_changed.emit())

    # --- API --------------------------------------------------------------
    @property
    def mode(self) -> str:
        return "raw" if self.mode_box.currentText() == "Cruda" else "processed"

    def set_data(self, data: np.ndarray | None, fs: float, channel_names: list[str]) -> None:
        self._data = data
        self._fs = fs
        self._channel_names = channel_names
        self._redraw()

    def clear(self) -> None:
        self._data = None
        self.plot.clear()
        self.add_seg_btn.setEnabled(False)
        self.sel_label.setText("Selección: —")

    def _gain(self) -> float:
        return float(self.gain_box.currentText().replace("x", ""))

    def _redraw(self) -> None:
        self.plot.clear()
        if self._data is None or self._data.size == 0:
            self.add_seg_btn.setEnabled(False)
            return

        data = self._data
        n_ch, n = data.shape
        t = np.arange(n) / self._fs

        # Normalización por canal solo para la visualización (no altera datos).
        disp = data.astype(np.float64)
        if self.norm_chk.isChecked():
            std = disp.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            disp = (disp - disp.mean(axis=1, keepdims=True)) / std
            self._spacing = 4.0
        else:
            self._spacing = np.nanmax(np.abs(disp)) * 1.2 + 1e-6

        gain = self._gain()
        ticks = []
        for i in range(n_ch):
            offset = (n_ch - 1 - i) * self._spacing
            curve = disp[i] * gain + offset
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            self.plot.plot(t, curve, pen=pg.mkPen(color, width=1))
            name = self._channel_names[i] if i < len(self._channel_names) else f"ch{i}"
            ticks.append((offset, name))

        axis = self.plot.getAxis("left")
        axis.setTicks([ticks])
        self.plot.setLabel("left", "Canal")

        self.plot.addItem(self.region)
        # Coloca la región en el primer 10% si no estaba visible.
        if not self.region.isVisible():
            self.region.setRegion((0, max(t[-1] * 0.1, t[1] if n > 1 else 0.1)))
            self.region.show()
        self.plot.setXRange(0, t[-1], padding=0.01)
        self.add_seg_btn.setEnabled(True)
        self._on_region_changed()

    # --- Selección --------------------------------------------------------
    def _on_region_changed(self) -> None:
        if self._data is None:
            return
        lo, hi = self.region.getRegion()
        lo = max(0.0, lo)
        s0 = int(round(lo * self._fs))
        s1 = int(round(hi * self._fs))
        s0, s1 = sorted((s0, s1))
        s1 = min(s1, self._data.shape[1])
        self.sel_label.setText(
            f"Selección: {lo:.2f}–{hi:.2f} s  ({s1 - s0} muestras)"
        )

    def selection_samples(self) -> tuple[int, int]:
        lo, hi = self.region.getRegion()
        s0 = int(round(max(0.0, lo) * self._fs))
        s1 = int(round(hi * self._fs))
        s0, s1 = sorted((s0, s1))
        return s0, min(s1, self._data.shape[1] if self._data is not None else s1)

    def _emit_segment(self) -> None:
        if self._data is None:
            return
        s0, s1 = self.selection_samples()
        if s1 - s0 < 2:
            return
        self.segment_requested.emit(s0, s1)
