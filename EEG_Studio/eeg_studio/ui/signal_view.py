"""Visor multicanal de señales EEG basado en pyqtgraph.

Muestra los canales apilados con desplazamiento vertical, permite alternar entre
señal cruda y procesada, navegar en el tiempo y seleccionar una región para
crear un segmento etiquetado (agrupar/aislar señales para el dataset).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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

# Paleta para colorear los segmentos por clase (color estable por etiqueta).
_SEGMENT_PALETTE = [
    "#66BB6A", "#42A5F5", "#FFA726", "#EC407A", "#AB47BC",
    "#26C6DA", "#D4E157", "#FF7043", "#8D6E63", "#5C6BC0",
]


def segment_color(label: str) -> str:
    """Color estable (mismo para la misma etiqueta) tomado de la paleta."""
    idx = int(hashlib.md5(str(label).encode("utf-8")).hexdigest(), 16) % len(_SEGMENT_PALETTE)
    return _SEGMENT_PALETTE[idx]


class SignalView(QWidget):
    segment_requested = pyqtSignal(int, int)  # (start_sample, stop_sample)
    mode_changed = pyqtSignal()               # cambió Cruda/Procesada

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None       # (n_canales, n_muestras)
        self._fs: float = 128.0
        self._channel_names: list[str] = []
        self._spacing: float = 0.0
        self._markers: list[tuple[int, str]] = []   # (muestra, etiqueta) — ayuda visual
        self._segments: list[tuple[int, int, str]] = []  # (inicio, fin, etiqueta)
        self._marker_items: list = []               # ítems del overlay (se reciclan)
        self._segment_items: list = []
        self._n = 0                                  # nº de muestras
        self._seg_top_y = 0.0                        # y para las etiquetas de segmento
        self._drawing = False                        # evita recursión al fijar el rango

        self._overlay_timer = QTimer(self)           # agrupa redibujos al hacer pan/zoom
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(40)
        self._overlay_timer.timeout.connect(self._redraw_overlay)

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
        self.gain_box.setToolTip(
            "Ganancia: amplificación SOLO visual de la señal (x2 = el doble de "
            "alta en pantalla). No modifica los datos; sirve para ver mejor "
            "fluctuaciones pequeñas o evitar que señales grandes se solapen."
        )
        self.gain_box.currentTextChanged.connect(self._redraw)
        controls.addWidget(QLabel("Ganancia:"))
        controls.addWidget(self.gain_box)

        self.norm_chk = QCheckBox("Normalizar vista")
        self.norm_chk.setChecked(True)
        self.norm_chk.stateChanged.connect(self._redraw)
        controls.addWidget(self.norm_chk)

        self.markers_chk = QCheckBox("Marcadores")
        self.markers_chk.setChecked(True)
        self.markers_chk.setToolTip(
            "Muestra los marcadores (Event Id) sobre la señal como ayuda visual "
            "para etiquetar manualmente las regiones de interés."
        )
        self.markers_chk.stateChanged.connect(self._redraw_overlay)
        controls.addWidget(self.markers_chk)

        self.segments_chk = QCheckBox("Segmentos")
        self.segments_chk.setChecked(True)
        self.segments_chk.setToolTip(
            "Sombrea los segmentos ya etiquetados sobre la señal, con un color "
            "por clase, para ver de un vistazo qué tramos ya están etiquetados."
        )
        self.segments_chk.stateChanged.connect(self._redraw_overlay)
        controls.addWidget(self.segments_chk)

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
        # Redibuja solo los marcadores/segmentos visibles al hacer pan/zoom.
        self.plot.getViewBox().sigXRangeChanged.connect(self._on_xrange_changed)
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

    def set_markers(self, markers: list[tuple[int, str]]) -> None:
        """Marcadores ``(muestra, etiqueta)`` a dibujar sobre la señal.

        Solo los almacena; el redibujado lo hace :meth:`set_data` a continuación.
        """
        self._markers = list(markers)

    def set_segments(self, segments: list[tuple[int, int, str]]) -> None:
        """Segmentos ``(inicio, fin, etiqueta)`` a sombrear sobre la señal."""
        self._segments = list(segments)

    def clear(self) -> None:
        self._data = None
        self.plot.clear()
        self.add_seg_btn.setEnabled(False)
        self.sel_label.setText("Selección: —")

    def _gain(self) -> float:
        return float(self.gain_box.currentText().replace("x", ""))

    def _redraw(self) -> None:
        self._drawing = True
        self.plot.clear()
        self._marker_items = []
        self._segment_items = []
        if self._data is None or self._data.size == 0:
            self.add_seg_btn.setEnabled(False)
            self._drawing = False
            return

        data = self._data
        n_ch, n = data.shape
        self._n = n
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
        self._seg_top_y = (n_ch - 0.4) * self._spacing

        self.plot.addItem(self.region)
        total = float(t[-1]) if n > 1 else 1.0
        # Ventana inicial legible: si la grabación es larga y hay muchos marcadores,
        # se muestra un tramo en torno al primer marcador (no los ~45 min de golpe).
        if not self.region.isVisible():
            if total > 60 and len(self._markers) > 30:
                start = max(0.0, min(s for s, _ in self._markers) / self._fs - 2.0)
                x0, x1 = start, min(total, start + 30.0)
            else:
                x0, x1 = 0.0, total
            self.region.setRegion((x0, x0 + min(4.0, (x1 - x0) * 0.5)))
            self.region.show()
            self.plot.setXRange(x0, x1, padding=0.0)
        self.add_seg_btn.setEnabled(True)
        self._drawing = False
        self._redraw_overlay()
        self._on_region_changed()

    def _on_xrange_changed(self, *_):
        if not self._drawing and self._data is not None:
            self._overlay_timer.start()          # agrupa redibujos del overlay

    def _visible_x(self) -> tuple[float, float]:
        try:
            (x0, x1), _ = self.plot.getViewBox().viewRange()
            return float(x0), float(x1)
        except Exception:  # noqa: BLE001
            return 0.0, self._n / self._fs

    def _redraw_overlay(self, *_) -> None:
        """Redibuja SOLO los marcadores/segmentos visibles en el rango actual."""
        for it in self._marker_items + self._segment_items:
            self.plot.removeItem(it)
        self._marker_items = []
        self._segment_items = []
        if self._data is None:
            return
        x0, x1 = self._visible_x()
        self._draw_segments(x0, x1)
        self._draw_markers(x0, x1)

    def _draw_segments(self, x0: float, x1: float) -> None:
        """Sombrea los segmentos visibles, con color por clase."""
        if not self.segments_chk.isChecked() or not self._segments:
            return
        fs, n = self._fs, self._n
        vis = [(a, b, l) for (a, b, l) in self._segments
               if max(0, a) / fs < x1 and min(b, n) / fs > x0]
        show_labels = len(vis) <= 40
        for start, stop, label in vis:
            t0 = max(0, start) / fs
            t1 = min(stop, n) / fs
            if t1 <= t0:
                continue
            color = segment_color(label)
            fill = pg.mkColor(color); fill.setAlpha(45)
            edge = pg.mkColor(color); edge.setAlpha(130)
            tip = (f"Clase: {label}\nRango: {t0:.2f}–{t1:.2f} s\n"
                   f"Muestras: {int(start)}–{int(stop)}  ({int(stop) - int(start)})")
            band = pg.LinearRegionItem(values=(t0, t1), brush=pg.mkBrush(fill),
                                       pen=pg.mkPen(edge), movable=False)
            band.setZValue(-10)
            band.setToolTip(tip)
            self.plot.addItem(band)
            self._segment_items.append(band)
            if show_labels:
                txt = pg.TextItem(str(label), color=color, anchor=(0, 1))
                txt.setPos(t0, self._seg_top_y)
                txt.setZValue(6)
                txt.setToolTip(tip)
                self.plot.addItem(txt)
                self._segment_items.append(txt)

    def _draw_markers(self, x0: float, x1: float) -> None:
        """Dibuja los marcadores visibles: atenuados, con etiqueta solo si hay pocos."""
        if not self.markers_chk.isChecked() or not self._markers:
            return
        fs, n = self._fs, self._n
        vis = [(s, l) for (s, l) in self._markers if 0 <= s <= n and x0 <= s / fs <= x1]
        if not vis:
            return
        if len(vis) > 300:                       # submuestrea para no dibujar miles
            vis = vis[:: len(vis) // 300 + 1]
        show_labels = len(vis) <= 25
        col = pg.mkColor("#FFD54F")
        col.setAlpha(170 if show_labels else 70)  # tenue cuando hay muchos
        pen = pg.mkPen(col, width=1, style=Qt.PenStyle.DashLine)
        for sample, label in vis:
            opts = {}
            if show_labels:
                opts = {"label": str(label), "labelOpts": {
                    "position": 0.93, "color": "#FFE082", "rotateAxis": (1, 0),
                    "fill": (20, 20, 20, 180)}}
            line = pg.InfiniteLine(pos=sample / fs, angle=90, pen=pen, **opts)
            line.setZValue(5)
            self.plot.addItem(line)
            self._marker_items.append(line)

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
