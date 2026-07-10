"""Visor en vivo: gráfico multicanal rodante para la adquisición en tiempo real.

Mantiene un buffer circular ``(n_canales, ventana)`` y actualiza curvas
persistentes (sin recrearlas) en cada refresco, para que sea fluido a ~30 fps.

Dos modos de **escala** (seleccionables):

* **Fija (µV)** — escala constante en microvoltios (estilo OpenViBE): la señal se
  dibuja a µV reales (quitando su offset DC) y las amplitudes son comparables y
  **no cambian solas**. Ajustable con «µV/canal».
* **Auto (normalizada)** — cada canal se normaliza por su desviación en la ventana
  (amplitud uniforme); cómodo pero la escala "respira".

Además permite **aislar un canal** para verlo solo, con sus medidas en vivo.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .signal_view import channel_color

# Opciones de µV por canal para la escala fija.
_UV_OPTIONS = ("20", "50", "100", "200", "500", "1000", "2000")


class LiveSignalView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buffer: np.ndarray | None = None
        self._fs = 128.0
        self._win = 640
        self._channels: list[str] = []
        self._curves: list[pg.PlotDataItem] = []
        self._x: np.ndarray = np.zeros(0)       # eje base [-ventana, 0] (relativo, cacheado)
        self._elapsed = 0                        # nº total de muestras recibidas (para el reloj)
        self._spacing = 4.0                     # separación (unidades z) en modo auto

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        # Aislar un canal.
        controls.addWidget(QLabel("Canal:"))
        self.channel_box = QComboBox()
        self.channel_box.addItem("Todos")
        self.channel_box.setToolTip("Aísla un canal para verlo solo y ver sus medidas en vivo.")
        self.channel_box.currentIndexChanged.connect(self._on_channel_changed)
        controls.addWidget(self.channel_box)

        # Escala: fija (µV) o auto (normalizada).
        controls.addSpacing(12)
        controls.addWidget(QLabel("Escala:"))
        self.scale_box = QComboBox()
        self.scale_box.addItems(["Fija (µV)", "Auto (normalizada)"])
        self.scale_box.setToolTip(
            "Fija: escala en µV constante (estilo OpenViBE), no cambia sola.\n"
            "Auto: cada canal se normaliza por su desviación (amplitud uniforme).")
        self.scale_box.currentIndexChanged.connect(self._on_scale_changed)
        controls.addWidget(self.scale_box)

        self.uv_box = QComboBox()
        self.uv_box.addItems(_UV_OPTIONS)
        self.uv_box.setCurrentText("200")
        self.uv_box.setToolTip("Microvoltios por canal en modo de escala fija.")
        self.uv_box.currentIndexChanged.connect(self._on_scale_changed)
        controls.addWidget(QLabel("µV/canal:"))
        controls.addWidget(self.uv_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setClipToView(True)
        layout.addWidget(self.plot)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #9be7c4; font-size: 11px;")
        self.stats_label.setVisible(False)
        layout.addWidget(self.stats_label)

    # --- Configuración -----------------------------------------------------
    def configure(self, channel_names: list[str], fs: float, window_seconds: float = 5.0) -> None:
        self._fs = float(fs)
        self._channels = list(channel_names)
        self._win = max(64, int(window_seconds * self._fs))
        self._elapsed = 0                        # reinicia el reloj del eje de tiempo
        n = len(channel_names)
        self._buffer = np.zeros((n, self._win), dtype=np.float64)

        self.plot.clear()
        self._curves = []
        # Eje de tiempo cacheado: es constante, no hace falta recrearlo cada frame.
        self._x = np.linspace(-self._win / self._fs, 0.0, self._win)
        for i in range(n):
            color = channel_color(channel_names[i], i)   # código de colores por región
            curve = self.plot.plot(self._x, np.zeros(self._win), pen=pg.mkPen(color, width=1))
            self._curves.append(curve)
        self.plot.setXRange(self._x[0], 0.0, padding=0.01)

        # Repuebla el selector de canales conservando la selección si sigue existiendo.
        current = self.channel_box.currentText()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItem("Todos")
        self.channel_box.addItems(list(channel_names))
        idx = self.channel_box.findText(current)
        self.channel_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_box.blockSignals(False)
        self._on_channel_changed()

    # --- Estado de los selectores -----------------------------------------
    def _isolated_index(self) -> int | None:
        i = self.channel_box.currentIndex()
        return (i - 1) if i > 0 else None

    def _scale_fixed(self) -> bool:
        return self.scale_box.currentIndex() == 0

    def _spacing_uv(self) -> float:
        try:
            return float(self.uv_box.currentText())
        except ValueError:
            return 200.0

    def _on_scale_changed(self, *_) -> None:
        self.uv_box.setEnabled(self._scale_fixed())
        self._apply_view_scale()
        if self._buffer is not None:
            self._redraw()

    def _on_channel_changed(self, *_) -> None:
        """Alterna entre vista multicanal apilada y un solo canal."""
        n = len(self._channels)
        iso = self._isolated_index()
        if iso is None or iso >= n:
            for c in self._curves:
                c.show()
            self.stats_label.setVisible(False)
        else:
            for i, c in enumerate(self._curves):
                c.setVisible(i == iso)
            self.stats_label.setVisible(True)
        self._apply_view_scale()
        if self._buffer is not None:
            self._redraw()

    def _apply_view_scale(self) -> None:
        """Fija ejes, etiquetas y rango Y según (aislado?, escala fija/auto)."""
        n = len(self._channels)
        vb = self.plot.getViewBox()
        iso = self._isolated_index()

        if iso is not None and iso < n:                       # un solo canal
            self.plot.getAxis("left").setTicks(None)          # eje µV numérico
            self.plot.setLabel("left", self._channels[iso], units="µV")
            # Fija: rango de altura constante (se recoloca en _redraw); Auto: autorango.
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=not self._scale_fixed())
            return

        # Multicanal apilado.
        fixed = self._scale_fixed()
        spacing = self._spacing_uv() if fixed else self._spacing
        ticks = [((n - 1 - i) * spacing, self._channels[i]) for i in range(n)]
        self.plot.getAxis("left").setTicks([ticks])
        if fixed:
            uv = self._spacing_uv()
            self.plot.setLabel("left", f"Canal  ·  {uv:g} µV entre canales")
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self.plot.setYRange(-0.75 * uv, (n - 1) * uv + 0.75 * uv, padding=0.0)
        else:
            self.plot.setLabel("left", "Canal (auto)")
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    # --- Datos -------------------------------------------------------------
    def append(self, chunk: np.ndarray) -> None:
        if self._buffer is None or chunk is None or chunk.size == 0:
            return
        k = chunk.shape[1]
        if k >= self._win:
            self._buffer = chunk[:, -self._win:].astype(np.float64)
        else:
            self._buffer = np.roll(self._buffer, -k, axis=1)
            self._buffer[:, -k:] = chunk
        self._elapsed += k                       # avanza el reloj del eje de tiempo
        self._redraw()

    def _redraw(self) -> None:
        buf = self._buffer
        n = buf.shape[0]
        base = self._x                          # eje base [-ventana, 0] (cacheado)
        if base.shape[0] != buf.shape[1]:       # salvaguarda si aún no se configuró
            base = np.linspace(-buf.shape[1] / self._fs, 0.0, buf.shape[1])
        # El eje avanza con el tiempo transcurrido: muestra [t_ahora-ventana, t_ahora].
        t_now = self._elapsed / self._fs
        x = base + t_now
        self.plot.setXRange(x[0], x[-1], padding=0.0)
        fixed = self._scale_fixed()

        iso = self._isolated_index()
        if iso is not None and iso < n:                       # un solo canal, a µV reales
            ch = buf[iso]
            self._curves[iso].setData(x, ch)
            mn, mx = float(ch.min()), float(ch.max())
            self.stats_label.setText(
                f"{self._channels[iso]}   ·   mín {mn:.1f}   ·   máx {mx:.1f}   ·   "
                f"media {float(ch.mean()):.1f}   ·   σ {float(ch.std()):.1f}   ·   "
                f"rango pico-a-pico {mx - mn:.1f} µV")
            if fixed:                                         # ventana de altura fija
                uv = self._spacing_uv()
                c = float(ch.mean())
                self.plot.setYRange(c - uv, c + uv, padding=0.0)
            return

        mean = buf.mean(axis=1, keepdims=True)
        if fixed:                                             # µV reales, sin renormalizar
            spacing = self._spacing_uv()
            disp = buf - mean                                 # quita el offset DC
        else:                                                 # z-score por canal (auto)
            std = buf.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            disp = (buf - mean) / std
            spacing = self._spacing
        for i in range(n):
            offset = (n - 1 - i) * spacing
            self._curves[i].setData(x, disp[i] + offset)

    def clear(self) -> None:
        if self._buffer is not None:
            self._buffer[:] = 0.0
            self._redraw()
