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
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .flow_layout import FlowLayout

pg.setConfigOptions(antialias=False, background="#101317", foreground="#c8d0d8")

_CURVE_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8", "#4DD0E1",
    "#AED581", "#FFD54F", "#F06292", "#9575CD", "#4DB6AC", "#DCE775",
    "#FF8A65", "#90A4AE",
]

# Código de colores por REGIÓN del cuero cabelludo (Emotiv EPOC+), con DOS TONOS
# por zona como el gorro de referencia:
#   AZUL frontal   → AF3/AF4 más fuerte, F7/F8 más claro
#   ROJO central   → F3/F4 oscuro (vino), FC5/FC6/T7/T8 salmón
#   VERDE post.    → P7/P8 más claro, O1/O2 más oscuro
_C_AF = "#1f5fd6"     # azul fuerte (AF3/AF4)
_C_F = "#5b9bf0"      # azul claro (F7/F8)
_C_F34 = "#a51d1d"    # rojo oscuro/vino (F3/F4)
_C_FCT = "#e46b60"    # rojo salmón (FC5/FC6/T7/T8)
_C_P = "#5cc274"      # verde claro (P7/P8)
_C_O = "#2e8f4f"      # verde oscuro (O1/O2)
_REGION_COLOR = {
    "AF3": _C_AF, "AF4": _C_AF,
    "F7": _C_F, "F8": _C_F,
    "F3": _C_F34, "F4": _C_F34,
    "FC5": _C_FCT, "FC6": _C_FCT, "T7": _C_FCT, "T8": _C_FCT,
    "P7": _C_P, "P8": _C_P,
    "O1": _C_O, "O2": _C_O,
}


def channel_color(name: str, idx: int) -> str:
    """Color de la curva de un canal: por REGIÓN si el nombre es de un electrodo EPOC+
    conocido (código de colores frontal/central/occipital); si no, la paleta cíclica."""
    key = str(name).strip().upper()
    return _REGION_COLOR.get(key, _CURVE_COLORS[idx % len(_CURVE_COLORS)])

# Paleta para colorear los segmentos por clase (color estable por etiqueta).
_SEGMENT_PALETTE = [
    "#66BB6A", "#42A5F5", "#FFA726", "#EC407A", "#AB47BC",
    "#26C6DA", "#D4E157", "#FF7043", "#8D6E63", "#5C6BC0",
]


def segment_color(label: str) -> str:
    """Color estable (mismo para la misma etiqueta) tomado de la paleta."""
    idx = int(hashlib.md5(str(label).encode("utf-8")).hexdigest(), 16) % len(_SEGMENT_PALETTE)
    return _SEGMENT_PALETTE[idx]


def _chip(*widgets) -> QWidget:
    """Agrupa una etiqueta y su(s) control(es) en un contenedor compacto para que
    NO se separen al reacomodarse la barra de controles (FlowLayout)."""
    box = QWidget()
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(3)
    for w in widgets:
        h.addWidget(w)
    return box


class SignalView(QWidget):
    segment_requested = pyqtSignal(int, int)  # (start_sample, stop_sample)
    cut_requested = pyqtSignal(int, int)      # recortar (eliminar) el tramo seleccionado
    delete_segments_requested = pyqtSignal(int, int)  # borrar segmentos de la selección
    relabel_segment_requested = pyqtSignal(str)   # reetiquetar un segmento (por id)
    delete_segment_requested = pyqtSignal(str)    # eliminar un segmento (por id)
    generate_periodic_requested = pyqtSignal(str)  # repetir un segmento periódicamente
    mode_changed = pyqtSignal()               # cambió Cruda/Procesada

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None       # (n_canales, n_muestras)
        self._fs: float = 128.0
        self._channel_names: list[str] = []
        self._spacing: float = 0.0
        self._markers: list[tuple[int, str]] = []   # (muestra, etiqueta) — ayuda visual
        self._segments: list[tuple[int, int, str]] = []  # (inicio, fin, etiqueta)
        self._cuts: list[tuple[int, int]] = []      # tramos eliminados (inicio, fin)
        self._marker_items: list = []               # ítems del overlay (se reciclan)
        self._segment_items: list = []
        self._cut_items: list = []
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

        # Todos los controles van en un FlowLayout: se reacomodan en tantas filas
        # como haga falta según el ancho, en vez de recortarse/desbordar cuando hay
        # muchos. Cada etiqueta va pegada a su control en un "chip" para que no se
        # separen al envolver. Un botón alterna entre barra compacta y expandida.
        flow = FlowLayout(h_spacing=8, v_spacing=4)

        self.expand_btn = QPushButton("⤢")
        self.expand_btn.setCheckable(True)
        self.expand_btn.setFixedWidth(30)
        self.expand_btn.setToolTip(
            "Expandir o compactar la barra de controles (ver todos los botones a "
            "la vez en varias filas, o dejarla compacta con desplazamiento)."
        )
        self.expand_btn.toggled.connect(self._toggle_controls_expanded)
        flow.addWidget(self.expand_btn)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Procesada", "Cruda"])
        self.mode_box.setToolTip("Mostrar la señal con o sin el pipeline aplicado")
        flow.addWidget(_chip(QLabel("Vista:"), self.mode_box))

        self.gain_box = QComboBox()
        self.gain_box.addItems(["x0.25", "x0.5", "x1", "x2", "x4", "x8"])
        self.gain_box.setCurrentText("x1")
        self.gain_box.setToolTip(
            "Ganancia: amplificación SOLO visual de la señal (x2 = el doble de "
            "alta en pantalla). No modifica los datos; sirve para ver mejor "
            "fluctuaciones pequeñas o evitar que señales grandes se solapen."
        )
        self.gain_box.currentTextChanged.connect(self._redraw)
        flow.addWidget(_chip(QLabel("Ganancia:"), self.gain_box))

        # Aislar un canal: muestra solo ese y sus medidas (rango de variación).
        self.channel_box = QComboBox()
        self.channel_box.addItem("Todos")
        self.channel_box.setToolTip("Aísla un canal para verlo solo y ver sus medidas.")
        self.channel_box.currentIndexChanged.connect(self._redraw)
        flow.addWidget(_chip(QLabel("Canal:"), self.channel_box))

        self.norm_chk = QCheckBox("Normalizar vista")
        self.norm_chk.setChecked(True)
        self.norm_chk.stateChanged.connect(self._redraw)
        flow.addWidget(self.norm_chk)

        self.markers_chk = QCheckBox("Marcadores")
        self.markers_chk.setChecked(True)
        self.markers_chk.setToolTip(
            "Muestra los marcadores (Event Id) sobre la señal como ayuda visual "
            "para etiquetar manualmente las regiones de interés."
        )
        self.markers_chk.stateChanged.connect(self._redraw_overlay)
        flow.addWidget(self.markers_chk)

        self.segments_chk = QCheckBox("Segmentos")
        self.segments_chk.setChecked(True)
        self.segments_chk.setToolTip(
            "Sombrea los segmentos ya etiquetados sobre la señal, con un color "
            "por clase, para ver de un vistazo qué tramos ya están etiquetados."
        )
        self.segments_chk.stateChanged.connect(self._redraw_overlay)
        flow.addWidget(self.segments_chk)

        self.sel_label = QLabel("Selección: —")
        flow.addWidget(self.sel_label)

        # Longitud (en tiempo) de la región seleccionada: fijarla a un valor exacto.
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setDecimals(2)
        self.len_spin.setRange(0.05, 3600.0)
        self.len_spin.setSingleStep(0.5)
        self.len_spin.setValue(4.0)
        self.len_spin.setSuffix(" s")
        self.len_spin.setMaximumWidth(90)
        self.len_spin.setEnabled(False)
        self.len_spin.setToolTip(
            "Fija la longitud (en tiempo) de la región seleccionada. Se ajusta "
            "manteniendo el inicio; útil para marcar ventanas de duración exacta."
        )
        self.len_spin.valueChanged.connect(self._on_length_changed)
        flow.addWidget(_chip(QLabel("Long.:"), self.len_spin))

        self.add_seg_btn = QPushButton("Crear segmento")
        self.add_seg_btn.setToolTip("Crea un segmento etiquetado a partir de la región seleccionada.")
        self.add_seg_btn.clicked.connect(self._emit_segment)
        self.add_seg_btn.setEnabled(False)
        flow.addWidget(self.add_seg_btn)

        # Edición de la señal sobre la selección.
        self.del_seg_btn = QPushButton("Borrar segmentos")
        self.del_seg_btn.setToolTip("Elimina los segmentos etiquetados que caen en la selección.")
        self.del_seg_btn.clicked.connect(self._emit_delete_segments)
        self.del_seg_btn.setEnabled(False)
        flow.addWidget(self.del_seg_btn)

        self.cut_btn = QPushButton("Recortar")
        self.cut_btn.setToolTip("Marca el tramo seleccionado como ELIMINADO (se excluye del "
                                "dataset y se sombrea en gris). No borra el CSV; reversible con Ctrl+Z.")
        self.cut_btn.clicked.connect(self._emit_cut)
        self.cut_btn.setEnabled(False)
        flow.addWidget(self.cut_btn)

        # --- Escalas de los ejes (rango X en tiempo, rango Y en amplitud) ---
        self.xstart_spin = QDoubleSpinBox()
        self.xstart_spin.setRange(0.0, 1e7); self.xstart_spin.setDecimals(2)
        self.xstart_spin.setSuffix(" s"); self.xstart_spin.setMaximumWidth(90)
        self.xstart_spin.setToolTip("Inicio de la ventana de tiempo visible.")
        self.xstart_spin.valueChanged.connect(self._apply_x_range)
        self.xwin_spin = QDoubleSpinBox()
        self.xwin_spin.setRange(0.05, 1e7); self.xwin_spin.setDecimals(2)
        self.xwin_spin.setValue(10.0); self.xwin_spin.setSuffix(" s")
        self.xwin_spin.setMaximumWidth(90)
        self.xwin_spin.setToolTip("Ancho de la ventana de tiempo (zoom horizontal).")
        self.xwin_spin.valueChanged.connect(self._apply_x_range)
        flow.addWidget(_chip(QLabel("Ejes · X desde"), self.xstart_spin,
                             QLabel("ventana"), self.xwin_spin))

        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9); self.ymin_spin.setDecimals(2)
        self.ymin_spin.setMaximumWidth(90)
        self.ymin_spin.setToolTip("Límite inferior del eje de amplitud.")
        self.ymin_spin.valueChanged.connect(self._apply_y_range)
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1e9, 1e9); self.ymax_spin.setDecimals(2)
        self.ymax_spin.setMaximumWidth(90)
        self.ymax_spin.setToolTip("Límite superior del eje de amplitud.")
        self.ymax_spin.valueChanged.connect(self._apply_y_range)
        flow.addWidget(_chip(QLabel("Y:"), self.ymin_spin, QLabel("a"), self.ymax_spin))

        self.autoscale_btn = QPushButton("Auto (ajustar)")
        self.autoscale_btn.setToolTip("Ajusta X e Y automáticamente para ver toda la señal.")
        self.autoscale_btn.clicked.connect(self._autoscale_axes)
        flow.addWidget(self.autoscale_btn)

        controls_host = QWidget()
        controls_host.setLayout(flow)
        self._controls_scroll = QScrollArea()
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._controls_scroll.setWidget(controls_host)
        self._controls_scroll.setMinimumWidth(0)
        self._controls_compact_h = 84          # ~2 filas; el resto con scroll vertical
        self._controls_scroll.setMaximumHeight(self._controls_compact_h)
        layout.addWidget(self._controls_scroll)

        # Fila de medidas del canal aislado (rango de variación de la señal).
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #9be7c4; font-size: 11px;")
        self.stats_label.setVisible(False)
        layout.addWidget(self.stats_label)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumSize(60, 40)          # que el visor se pueda encoger
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        # Redibuja solo los marcadores/segmentos visibles al hacer pan/zoom.
        self.plot.getViewBox().sigXRangeChanged.connect(self._on_xrange_changed)
        # Refleja el rango actual en los campos de escala (al hacer pan/zoom con el ratón).
        self.plot.getViewBox().sigRangeChanged.connect(self._sync_axis_spins)
        # Clic derecho sobre un segmento: reetiquetar / eliminar.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)
        layout.addWidget(self.plot, 1)

        # Región de selección de tiempo.
        self.region = pg.LinearRegionItem(brush=(80, 160, 255, 40))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.hide()

        self.mode_box.currentTextChanged.connect(lambda *_: self.mode_changed.emit())

    def _toggle_controls_expanded(self, expanded: bool) -> None:
        """Expande la barra de controles (muestra todas las filas de golpe) o la
        deja compacta (con desplazamiento vertical si no caben)."""
        self._controls_scroll.setMaximumHeight(260 if expanded else self._controls_compact_h)
        self.expand_btn.setText("⤡" if expanded else "⤢")

    # --- API --------------------------------------------------------------
    @property
    def mode(self) -> str:
        return "raw" if self.mode_box.currentText() == "Cruda" else "processed"

    def set_data(self, data: np.ndarray | None, fs: float, channel_names: list[str]) -> None:
        self._data = data
        self._fs = fs
        self._channel_names = channel_names
        self._sync_channel_box(channel_names)
        self._redraw()

    def _sync_channel_box(self, names: list[str]) -> None:
        """Rellena el combo de canales conservando la selección si sigue existiendo."""
        current = self.channel_box.currentText()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItem("Todos")
        self.channel_box.addItems(list(names))
        idx = self.channel_box.findText(current)
        self.channel_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_box.blockSignals(False)

    def _isolated_index(self) -> int | None:
        """Índice del canal aislado, o None si está en «Todos»."""
        i = self.channel_box.currentIndex()
        return (i - 1) if i > 0 else None

    def set_markers(self, markers: list[tuple[int, str]]) -> None:
        """Marcadores ``(muestra, etiqueta)`` a dibujar sobre la señal.

        Solo los almacena; el redibujado lo hace :meth:`set_data` a continuación.
        """
        self._markers = list(markers)

    def set_segments(self, segments: list[tuple[int, int, str]]) -> None:
        """Segmentos ``(inicio, fin, etiqueta)`` a sombrear sobre la señal."""
        self._segments = list(segments)

    def set_cuts(self, cuts: list[tuple[int, int]]) -> None:
        """Tramos eliminados ``(inicio, fin)`` a sombrear en gris (excluidos)."""
        self._cuts = [(int(a), int(b)) for a, b in cuts]

    def clear(self) -> None:
        self._data = None
        self.plot.clear()
        self._set_edit_buttons(False)
        self.sel_label.setText("Selección: —")
        self.stats_label.setVisible(False)

    def _set_edit_buttons(self, on: bool) -> None:
        for b in (self.add_seg_btn, self.del_seg_btn, self.cut_btn):
            b.setEnabled(on)
        self.len_spin.setEnabled(on)

    def _gain(self) -> float:
        return float(self.gain_box.currentText().replace("x", ""))

    def _redraw(self) -> None:
        self._drawing = True
        self.plot.clear()
        self._marker_items = []
        self._segment_items = []
        self._cut_items = []
        if self._data is None or self._data.size == 0:
            self._set_edit_buttons(False)
            self._drawing = False
            return

        data = self._data
        n_ch, n = data.shape
        self._n = n
        t = np.arange(n) / self._fs

        # Canal aislado: solo ese canal, a escala real, con sus medidas.
        iso = self._isolated_index()
        if iso is not None and iso < n_ch:
            self._draw_isolated(iso, np.asarray(data[iso], dtype=np.float64), t, n)
            self._set_edit_buttons(True)
            self._drawing = False
            self._redraw_overlay()
            self._on_region_changed()
            return
        self.stats_label.setVisible(False)

        # Se resta la media por canal SOLO para visualizar (quita el offset DC —
        # p. ej. ~4200 µV del EPOC+): así la escala refleja la amplitud real y cada
        # canal se centra en su etiqueta. No altera los datos.
        disp = data.astype(np.float64)
        disp = disp - disp.mean(axis=1, keepdims=True)
        if self.norm_chk.isChecked():
            std = disp.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            disp = disp / std
            self._spacing = 4.0
        else:
            self._spacing = np.nanmax(np.abs(disp)) * 1.2 + 1e-6

        gain = self._gain()
        ticks = []
        for i in range(n_ch):
            offset = (n_ch - 1 - i) * self._spacing
            curve = disp[i] * gain + offset
            name = self._channel_names[i] if i < len(self._channel_names) else f"ch{i}"
            self.plot.plot(t, curve, pen=pg.mkPen(channel_color(name, i), width=1))
            ticks.append((offset, name))

        axis = self.plot.getAxis("left")
        axis.setTicks([ticks])
        self.plot.setLabel("left", "Canal")
        self._seg_top_y = (n_ch - 0.4) * self._spacing

        self._setup_region(float(t[-1]) if n > 1 else 1.0)
        self._set_edit_buttons(True)
        self._drawing = False
        self._redraw_overlay()
        self._on_region_changed()

    def _setup_region(self, total: float) -> None:
        """Coloca la región de selección y una ventana inicial legible."""
        self.plot.addItem(self.region)
        # La longitud máxima ajustable es la duración total de la señal.
        self.len_spin.blockSignals(True)
        self.len_spin.setMaximum(max(0.1, round(total, 3)))
        self.len_spin.blockSignals(False)
        # Si la grabación es larga y hay muchos marcadores, se muestra un tramo
        # en torno al primer marcador (no los ~45 min de golpe).
        if not self.region.isVisible():
            if total > 60 and len(self._markers) > 30:
                start = max(0.0, min(s for s, _ in self._markers) / self._fs - 2.0)
                x0, x1 = start, min(total, start + 30.0)
            else:
                x0, x1 = 0.0, total
            self.region.setRegion((x0, x0 + min(4.0, (x1 - x0) * 0.5)))
            self.region.show()
            self.plot.setXRange(x0, x1, padding=0.0)

    def _draw_isolated(self, idx: int, ch: np.ndarray, t: np.ndarray, n: int) -> None:
        """Dibuja un único canal a escala real y muestra sus medidas."""
        gain = self._gain()
        normalized = self.norm_chk.isChecked()
        disp = ch - ch.mean()          # centrar (quita el offset DC) para visualizar
        if normalized:
            s = float(ch.std()) or 1.0
            disp = disp / s
        name = self._channel_names[idx] if idx < len(self._channel_names) else f"ch{idx}"
        self.plot.plot(t, disp * gain, pen=pg.mkPen(channel_color(name, idx), width=1))
        self.plot.getAxis("left").setTicks(None)   # ticks numéricos automáticos
        self.plot.setLabel("left", name, units=None if normalized else "µV")

        # Medidas sobre los datos reales (rango de variación de la señal).
        mn, mx, mean, std = float(ch.min()), float(ch.max()), float(ch.mean()), float(ch.std())
        self.stats_label.setText(
            f"{name}   ·   mín {mn:.1f}   ·   máx {mx:.1f}   ·   media {mean:.1f}"
            f"   ·   σ {std:.1f}   ·   rango pico-a-pico {mx - mn:.1f} µV")
        self.stats_label.setVisible(True)

        top = float((disp * gain).max()) if disp.size else 1.0
        rng = float((disp * gain).max() - (disp * gain).min()) if disp.size else 1.0
        self._seg_top_y = top + rng * 0.02
        self._setup_region(float(t[-1]) if n > 1 else 1.0)

    def _on_xrange_changed(self, *_):
        if not self._drawing and self._data is not None:
            self._overlay_timer.start()          # agrupa redibujos del overlay

    # --- Escalas de los ejes ---------------------------------------------
    def _apply_x_range(self, *_):
        """Fija el rango del eje X (tiempo) desde los campos «desde/ventana»."""
        if self._data is None or self._drawing:
            return
        x0 = self.xstart_spin.value()
        w = max(self.xwin_spin.value(), 1e-3)
        self.plot.setXRange(x0, x0 + w, padding=0.0)

    def _apply_y_range(self, *_):
        """Fija el rango del eje Y (amplitud) desde los campos «min/max»."""
        if self._data is None or self._drawing:
            return
        lo, hi = self.ymin_spin.value(), self.ymax_spin.value()
        if hi > lo:
            self.plot.setYRange(lo, hi, padding=0.0)

    def _autoscale_axes(self):
        """Ajusta ambos ejes para ver toda la señal."""
        if self._data is None:
            return
        total = max(1e-3, self._data.shape[1] / self._fs)
        self.plot.getViewBox().autoRange()               # Y (y X) a los datos
        self.plot.setXRange(0.0, total, padding=0.02)    # X a toda la señal (determinista)
        self._sync_axis_spins()

    def _sync_axis_spins(self, *_):
        """Refleja el rango visible actual en los campos (sin re-disparar cambios)."""
        try:
            (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        except Exception:  # noqa: BLE001
            return
        for sp, val in ((self.xstart_spin, x0), (self.xwin_spin, x1 - x0),
                        (self.ymin_spin, y0), (self.ymax_spin, y1)):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)

    def _segment_at(self, sample: int):
        """Segmento cuyo ``[inicio, fin)`` contiene ``sample`` (el más específico).

        Devuelve ``(id, etiqueta)`` o ``None``. Necesita que los segmentos lleven
        su id (4º elemento de la tupla)."""
        hits = [s for s in self._segments
                if len(s) >= 4 and s[3] and s[0] <= sample < s[1]]
        if not hits:
            return None
        start, stop, label, seg_id = min(hits, key=lambda s: s[1] - s[0])[:4]
        return seg_id, label

    def _on_scene_clicked(self, ev) -> None:
        """Clic derecho sobre un segmento → menú Reetiquetar / Eliminar."""
        if ev.button() != Qt.MouseButton.RightButton or self._data is None:
            return
        try:
            pos = self.plot.getViewBox().mapSceneToView(ev.scenePos())
        except Exception:  # noqa: BLE001
            return
        hit = self._segment_at(int(round(pos.x() * self._fs)))
        if hit is None:
            return                                # no hay segmento aquí: menú por defecto
        ev.accept()
        seg_id, label = hit
        menu = QMenu()
        act_re = menu.addAction(f"Reetiquetar «{label}»…")
        act_gen = menu.addAction("Repetir periódicamente… (generar segmentos)")
        act_del = menu.addAction(f"Eliminar segmento «{label}»")
        chosen = menu.exec(QCursor.pos())
        if chosen is act_re:
            self.relabel_segment_requested.emit(seg_id)
        elif chosen is act_gen:
            self.generate_periodic_requested.emit(seg_id)
        elif chosen is act_del:
            self.delete_segment_requested.emit(seg_id)

    def _visible_x(self) -> tuple[float, float]:
        try:
            (x0, x1), _ = self.plot.getViewBox().viewRange()
            return float(x0), float(x1)
        except Exception:  # noqa: BLE001
            return 0.0, self._n / self._fs

    def _redraw_overlay(self, *_) -> None:
        """Redibuja SOLO los marcadores/segmentos visibles en el rango actual."""
        for it in self._marker_items + self._segment_items + self._cut_items:
            self.plot.removeItem(it)
        self._marker_items = []
        self._segment_items = []
        self._cut_items = []
        if self._data is None:
            return
        x0, x1 = self._visible_x()
        self._draw_segments(x0, x1)
        self._draw_cuts(x0, x1)
        self._draw_markers(x0, x1)

    def _draw_cuts(self, x0: float, x1: float) -> None:
        """Sombrea en gris los tramos eliminados (excluidos del dataset)."""
        if not self._cuts:
            return
        fs, n = self._fs, self._n
        for a, b in self._cuts:
            t0 = max(0, a) / fs
            t1 = min(b, n) / fs
            if t1 <= t0 or t0 > x1 or t1 < x0:
                continue
            band = pg.LinearRegionItem(values=(t0, t1),
                                       brush=pg.mkBrush(120, 124, 132, 120),
                                       pen=pg.mkPen(170, 174, 182, 200), movable=False)
            band.setZValue(20)                       # por encima de la señal
            band.setToolTip(f"Tramo eliminado: {t0:.2f}–{t1:.2f} s (excluido del dataset)")
            self.plot.addItem(band)
            self._cut_items.append(band)
            txt = pg.TextItem("eliminado", color="#dfe3ea", anchor=(0, 0))
            txt.setPos(t0, self._seg_top_y)
            txt.setZValue(21)
            self.plot.addItem(txt)
            self._cut_items.append(txt)

    def _draw_segments(self, x0: float, x1: float) -> None:
        """Sombrea los segmentos visibles, con color por clase."""
        if not self.segments_chk.isChecked() or not self._segments:
            return
        fs, n = self._fs, self._n
        vis = [s for s in self._segments
               if max(0, s[0]) / fs < x1 and min(s[1], n) / fs > x0]
        show_labels = len(vis) <= 40
        for seg in vis:
            start, stop, label = seg[0], seg[1], seg[2]
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
        # Refleja la longitud actual en el campo (sin re-disparar el cambio).
        length = max(0.0, hi - lo)
        self.len_spin.blockSignals(True)
        self.len_spin.setValue(min(length, self.len_spin.maximum()))
        self.len_spin.blockSignals(False)

    def _on_length_changed(self, seconds: float) -> None:
        """Fija la longitud (en tiempo) de la selección, manteniendo el inicio.

        Si no cabe hasta el final de la señal, corre el inicio hacia atrás para
        conservar la duración pedida."""
        if self._data is None or seconds <= 0:
            return
        total = self._data.shape[1] / self._fs
        lo, _hi = self.region.getRegion()
        lo = max(0.0, min(lo, total))
        hi = lo + seconds
        if hi > total:                       # no cabe: recorta al final y corre el inicio
            hi = total
            lo = max(0.0, total - seconds)
        self.region.setRegion((lo, hi))      # dispara _on_region_changed (etiqueta + campo)

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

    def _emit_cut(self) -> None:
        if self._data is None:
            return
        s0, s1 = self.selection_samples()
        if s1 - s0 < 1:
            return
        self.cut_requested.emit(s0, s1)

    def _emit_delete_segments(self) -> None:
        if self._data is None:
            return
        s0, s1 = self.selection_samples()
        self.delete_segments_requested.emit(s0, s1)
